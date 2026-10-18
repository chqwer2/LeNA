import torch
import torch.nn as nn
from typing import Optional, Literal

GateType = Literal["none", "sigmoid", "rezero"]
GateMode = Literal["global", "spatial", "channel", "voxel", "input", "context"]
GateStrength = Literal["soft", "hard"]


class Gate(nn.Module):
    """
    ViT-oriented gating with lazy shape inference.

    Expected input:
        x: (B, N, C)
          N = tokens
          C = embedding dim

    gate_mode:
      - global  : scalar
      - spatial : per-token (N)
      - channel : per-dim (C)
      - voxel   : per-(N,C)
    """

    def __init__(
        self,
        gate_type: GateType,
        gate_mode: GateMode,
        gate_strength: GateStrength,
        init: float,
        dtype: torch.dtype,
        device: torch.device,
        code_dim: Optional[int] = None,
    ):
        super().__init__()
        self.gate_type = gate_type
        self.gate_mode = gate_mode
        self.gate_strength = gate_strength
        self.init = float(init)  # honor configured gate_init (e.g. negative => start near-off)
        self.dtype = dtype
        self.device = device

        self.param: Optional[nn.Parameter] = None

        if gate_type == "none":
            return

        # The parameter MUST exist before the optimizer is built, otherwise the gate is
        # silently frozen at its init value and 'learn where nonlinearity is needed'
        # (plus the gate-L1 sparsity penalty) does nothing at all. Both shapes that do
        # not depend on the input sequence length are therefore built eagerly here.
        if gate_mode == "context":
            # g(z_i) = sigmoid(w.z_i + u.causal_mean(z_<=i) + b): the gate that decides
            # nonlinearity strength depends on the CAUSAL CONTEXT, not just the current
            # token. This is the cross-token piece every prior nonlinear-PEFT lacks --
            # AuroRA/LeNA gate/activate per token independently, which cannot model the
            # reasoning dependency ("what to compute at '?' depends on preceding tokens").
            # Causal (cumulative) mean is cheap and does not leak future tokens.
            if code_dim is None:
                raise ValueError("Gate(gate_mode='context') needs code_dim (the rank r)")
            self.w = nn.Parameter(torch.zeros(int(code_dim), dtype=dtype, device=device))
            self.u = nn.Parameter(torch.zeros(int(code_dim), dtype=dtype, device=device))
            self.param = nn.Parameter(torch.full((1,), self.init, dtype=dtype, device=device))
            self.register_buffer("obs_openness", torch.tensor(float("nan")))
        elif gate_mode == "input":
            # g(z) = sigmoid(w.z + b): the INPUT-CONDITIONAL gate the paper describes
            # (sec. 3.5) but the code never implemented -- it shipped a static
            # sigmoid(theta), which reviewers flagged as a code/paper mismatch.
            #
            # It also fixes the static gate's dead ends: a hard gate initialized closed
            # multiplies phi by exactly 0, so phi receives no gradient and the gate's own
            # gradient (measured at ~1/300 of B's) can never pry it open; initialized
            # open it just stays open everywhere. A per-token gate is a genuine router --
            # under multi-task training it can open for the tasks that need curvature.
            if code_dim is None:
                raise ValueError("Gate(gate_mode='input') needs code_dim (the rank r)")
            self.w = nn.Parameter(torch.zeros(int(code_dim), dtype=dtype, device=device))
            self.param = nn.Parameter(torch.full((1,), self.init, dtype=dtype, device=device))
            # running mean of the realized openness, for the where-map at eval time
            self.register_buffer("obs_openness", torch.tensor(float("nan")))
        elif gate_mode == "global":
            self.param = nn.Parameter(torch.full((1,), self.init, dtype=dtype, device=device))
        elif gate_mode == "channel":
            if code_dim is None:
                raise ValueError("Gate(gate_mode='channel') needs code_dim (the rank r)")
            self.param = nn.Parameter(
                torch.full((int(code_dim),), self.init, dtype=dtype, device=device)
            )
        else:
            # 'spatial'/'voxel' are sized by the token count, which is not known until the
            # first forward -- i.e. after the optimizer exists. Refuse instead of training
            # a gate that never receives an update.
            raise ValueError(
                f"gate_mode={gate_mode!r} is sized by sequence length and cannot be "
                "registered before the optimizer is built; use 'global' or 'channel'."
            )

    # -------------------------
    # lazy initialization
    # -------------------------
    def _init_param_from_x(self, x: torch.Tensor):
        # Kept only as a safety net: every supported gate_mode is now built eagerly in
        # __init__ so the parameter reaches the optimizer.
        if self.param is not None:
            return
        self.device = x.device

        if x.ndim != 3:
            raise ValueError(
                f"Gate expects ViT-like input (B,N,C), got shape {tuple(x.shape)}"
            )

        _, N, C = x.shape

        if self.gate_mode == "global":
            shape = (1,)

        elif self.gate_mode == "spatial":
            shape = (N,)

        elif self.gate_mode == "channel":
            shape = (C,)

        elif self.gate_mode == "voxel":
            shape = (N, C)

        else:
            raise ValueError(f"Unknown gate_mode: {self.gate_mode}")

        self.param = nn.Parameter(
            torch.full(shape, self.init, dtype=self.dtype, device=self.device)
        )




    # -------------------------
    # straight-through helpers
    # -------------------------
    def _hard_sigmoid_st(self, p: torch.Tensor) -> torch.Tensor:
        soft = torch.sigmoid(p)
        hard = (soft >= 0.5).to(soft.dtype)
        return hard.detach() - soft.detach() + soft

    def _hard_rezero_st(self, p: torch.Tensor) -> torch.Tensor:
        hard = (p > 0).to(p.dtype)
        return hard.detach() - p.detach() + p

    # -------------------------
    # reshape for broadcasting
    # -------------------------
    def _reshape_param_for_x(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, C)
        """
        p = self.param

        assert p is not None

        if self.gate_mode == "global":
            return p.view(1, 1, 1)

        if self.gate_mode == "spatial":
            return p.view(1, -1, 1)  # (1,N,1)

        if self.gate_mode == "channel":
            return p.view(1, 1, -1)  # (1,1,C)

        if self.gate_mode == "voxel":
            return p.view(1, *p.shape)  # (1,N,C)

        raise ValueError(f"Unknown gate_mode: {self.gate_mode}")

    # -------------------------
    # gate value (for interpolation h = z + g*(phi-z))
    # -------------------------
    def value(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """Return gate g in [0,1], reshaped to broadcast against x. None if gate disabled."""
        if self.gate_type == "none":
            return None
        if self.gate_mode == "context":
            # x is the code z, shape [B, T, C] (token dim = -2). Causal mean over tokens.
            xf = x
            tdim = -2
            csum = torch.cumsum(xf, dim=tdim)
            cnt = torch.arange(1, xf.shape[tdim] + 1, device=xf.device, dtype=xf.dtype)
            shape = [1] * xf.ndim; shape[tdim] = xf.shape[tdim]
            ctx = csum / cnt.view(shape)                       # causal_mean(z_<=i)
            logit = ((x * self.w.to(x.dtype)).sum(dim=-1, keepdim=True)
                     + (ctx * self.u.to(x.dtype)).sum(dim=-1, keepdim=True)
                     + self.param.to(x.dtype))
            g = torch.sigmoid(logit) if self.gate_strength == "soft" else self._hard_sigmoid_st(logit)
            self.last_openness = g.mean()
            with torch.no_grad():
                cur = g.detach().float().mean()
                self.obs_openness = cur if torch.isnan(self.obs_openness) else (0.99*self.obs_openness + 0.01*cur)
            return g
        if self.gate_mode == "input":
            logit = (x * self.w.to(x.dtype)).sum(dim=-1, keepdim=True) + self.param.to(x.dtype)
            g = torch.sigmoid(logit) if self.gate_strength == "soft" else self._hard_sigmoid_st(logit)
            # keep both a differentiable batch mean (for the L1 "learn where" penalty)
            # and a detached running mean (for the where-map report)
            self.last_openness = g.mean()
            with torch.no_grad():
                cur = g.detach().float().mean()
                self.obs_openness = cur if torch.isnan(self.obs_openness) else (
                    0.99 * self.obs_openness + 0.01 * cur)
            return g
        self._init_param_from_x(x)
        p = self._reshape_param_for_x(x)
        if self.gate_type == "sigmoid":
            return torch.sigmoid(p) if self.gate_strength == "soft" else self._hard_sigmoid_st(p)
        if self.gate_type == "rezero":
            return p if self.gate_strength == "soft" else self._hard_rezero_st(p)
        raise ValueError(f"Unknown gate_type: {self.gate_type}")

    def openness(self) -> Optional[torch.Tensor]:
        """Differentiable per-element gate openness in [0,1] from the raw parameter.
        Input-independent (the gate parameter does not depend on x), so usable directly
        for an L1 sparsity penalty and for 'where is nonlinearity used' analysis."""
        if self.gate_type == "none" or self.param is None:
            return None
        if self.gate_mode in ("input", "context"):
            # differentiable when a forward has run this step (L1 penalty), else the
            # running estimate (reporting)
            lo = getattr(self, "last_openness", None)
            if lo is not None:
                return lo.reshape(1)
            return None if torch.isnan(self.obs_openness) else self.obs_openness.reshape(1)
        if self.gate_type == "sigmoid":
            return torch.sigmoid(self.param)
        if self.gate_type == "rezero":
            return self.param.clamp(0.0, 1.0)
        return None

    # -------------------------
    # forward
    # -------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gate_type == "none":
            return x

        # infer N, C on first call
        self._init_param_from_x(x)

        p = self._reshape_param_for_x(x)

        if self.gate_type == "sigmoid":
            if self.gate_strength == "soft":
                gate = torch.sigmoid(p)
            elif self.gate_strength == "hard":
                gate = self._hard_sigmoid_st(p)
            else:
                raise ValueError(f"Unknown gate_strength: {self.gate_strength}")
            return x * gate

        if self.gate_type == "rezero":
            if self.gate_strength == "soft":
                gate = p
            elif self.gate_strength == "hard":
                gate = self._hard_rezero_st(p)
            else:
                raise ValueError(f"Unknown gate_strength: {self.gate_strength}")
            return x * gate

        raise ValueError(f"Unknown gate_type: {self.gate_type}")
