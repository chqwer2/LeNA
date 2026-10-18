from __future__ import annotations

from typing import Any, Literal, Optional, Tuple

import math
import torch
import torch.nn as nn
import math
import torch.nn.functional as F


FlexMode = Literal["global", "token", "dim", "voxel"]
ActKind = Literal["identity", "relu", "swish", "gelu", "fourier", "spline", "polynomial"]


# -----------------------
# Shape helpers
# -----------------------

def _infer_hwc(x: torch.Tensor) -> Tuple[int, int, int]:
    """
    Expect x shaped [..., H, W, C] (C last).
    """
    if x.ndim < 3:
        raise ValueError(f"Expected [...,H,W,C], got {tuple(x.shape)}")
    return int(x.shape[-3]), int(x.shape[-2]), int(x.shape[-1])


def _require_max_hw(mode: FlexMode, max_h: Optional[int], max_w: Optional[int]):
    """
    For spatial/voxel params, H/W can change (seq_len changes), so we must allocate
    at a fixed max and slice.
    """
    if mode in ("token", "voxel"):
        if max_h is None:
            raise ValueError(
                f"Flex mode '{mode}' requires max_h (and optionally max_w) to support variable H/W."
            )
        if max_w is None:
            # most transformer cases use W=1, so default to 1 if not specified
            max_w = 1
    return max_h, max_w


def _param_base_shape(
    mode: FlexMode,
    H: int,
    W: int,
    C: int,
    *,
    max_h: Optional[int] = None,
    max_w: Optional[int] = None,
) -> Tuple[int, int, int]:
    """
    Returns base parameter shape (H', W', C') before any extra dims (like terms/knots/degree).

    Semantics:
      - global:   (1, 1, 1)
      - channel:  (1, 1, C)     -> per-channel parameters (stable for transformers)
      - spatial:  (H, W, 1)     -> per-position parameters (requires max_h/max_w for variable H)
      - voxel:    (H, W, C)     -> per-position-per-channel (requires max_h/max_w for variable H)
    """
    if mode == "global":
        return (1, 1, 1)
    if mode == "dim":
        return (1, 1, C)
    if mode == "token":
        return (int(H), int(W), 1)
    if mode == "voxel":
        # max_h, max_w = _require_max_hw(mode, max_h, max_w)
        return (int(H), int(W), C)
    raise ValueError(f"Unknown flex mode: {mode}")


def _broadcast_param_to_x(p: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Make p broadcastable to x's ndim.
    x is [..., H, W, C]
    p is [H', W', C'] or [H', W', C', extra...] (after we append extra dims).
    We'll add leading singleton dims until it matches x.ndim (or x.ndim+1 for extra dims cases).
    """
    while p.ndim < x.ndim:
        p = p.unsqueeze(0)
    return p


def _slice_hw(p: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    Slice parameter table (max_h, max_w, ...) to current (H,W).
    Assumes p has at least 2 dims and first two are H/W axes.
    """
    if p.shape[0] < H or p.shape[1] < W:
        raise ValueError(
            f"Input H,W=({H},{W}) exceed parameter table size ({p.shape[0]},{p.shape[1]}). "
            "Increase max_h/max_w."
        )
    return p[:H, :W, ...]


# -----------------------
# Activations
# -----------------------

class IdentityAct(nn.Module):
    kind = "identity"
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class FlexReLU(nn.Module):
    kind = "relu"

    def __init__(self, mode: FlexMode, init_a: float = 0.25, max_h: Optional[int] = None, max_w: Optional[int] = None):
        super().__init__()
        self.mode = mode
        self.init_a = float(init_a)
        self.max_h = max_h
        self.max_w = max_w
        self.a: Optional[nn.Parameter] = None
        self._C: Optional[int] = None  # for channel/voxel consistency

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x >= 0, x, 0)



class FlexSwish(nn.Module):
    """
    "Flex" Swish/SiLU: y = x * sigmoid(beta * x)

    beta can be:
      - "dim"/"voxel": per-channel (C) (and possibly per-voxel depending on your _param_base_shape)
      - "token"/"voxel": optionally per-(H,W,...) and sliced with _slice_hw like your FlexFourier

    Optional gating for identity init:
      - "soft":     y = gate * swish(x) + (1-gate) * x
      - "hard":     same but hard gate (ST)
      - "residual": y = x + gate * (swish(x) - x)
      - "none":     y = swish(x)
    """
    kind = "flexswish"

    def __init__(
        self,
        mode: FlexMode = "dim",
        init_beta: float = 1.0,
        max_h: Optional[int] = None,
        max_w: Optional[int] = None,
        use_gate: str = "residual",   # ["soft", "hard", "none", "residual"]
        init_gate: float = -8.0,      # sigmoid(-8) ~ 0.0003 => near-identity at init
    ):
        super().__init__()
        self.mode = mode
        self.max_h = max_h
        self.max_w = max_w
        self.use_gate = use_gate

        self.init_beta = float(init_beta)
        self.init_gate = float(init_gate)

        self.beta: Optional[nn.Parameter] = None  # unconstrained, mapped via softplus -> positive
        self.t: Optional[nn.Parameter] = None     # gate logits
        self._C: Optional[int] = None

    def _maybe_init(self, x: torch.Tensor):
        H, W, C = _infer_hwc(x)

        if self.beta is None:
            base = _param_base_shape(self.mode, H, W, C, max_h=self.max_h, max_w=self.max_w)

            # softplus(u) ~= init_beta  => u ~= log(exp(init_beta)-1)
            init_beta_u = math.log(math.expm1(self.init_beta) + 1e-6)
            self.beta = nn.Parameter(torch.full(base, init_beta_u, device=x.device, dtype=x.dtype))

            self._C = C

            if self.use_gate != "none":
                self.t = nn.Parameter(
                    torch.full((H, W, C), self.init_gate, dtype=x.dtype, device=x.device)
                )

        else:
            if self.mode in ("dim", "voxel") and self._C is not None and C != self._C:
                raise ValueError(f"Channel size C changed from {self._C} to {C} for mode='{self.mode}'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._maybe_init(x)
        H, W, _ = _infer_hwc(x)

        beta = self.beta
        if beta is None:
            return F.silu(x)

        if self.mode in ("token", "voxel"):
            beta = _slice_hw(beta, H, W)
            t = _slice_hw(self.t, H, W) if self.t is not None else None
        else:
            t = self.t

        # Broadcast params up to x.ndim
        while beta.ndim < x.ndim:
            beta = beta.unsqueeze(0)
            if t is not None:
                t = t.unsqueeze(0)

        beta_pos = F.softplus(beta)
        swish = x * torch.sigmoid(beta_pos * x)

        if self.use_gate == "none":
            return swish

        if t is None:
            return swish

        if self.use_gate == "soft":
            gate = torch.sigmoid(t)
            return swish * gate + x * (1.0 - gate)

        if self.use_gate == "hard":
            gate = _hard_sigmoid_st(t)
            return swish * gate + x * (1.0 - gate)

        if self.use_gate == "residual":
            gate = torch.sigmoid(t)
            return x + gate * (swish - x)

        raise ValueError(f"Unknown use_gate='{self.use_gate}'. Expected one of ['soft','hard','none','residual'].")




class FlexGELU(nn.Module):
    kind = "gelu"

    def __init__(self, mode: FlexMode, init_k: float = 1.0, max_h: Optional[int] = None, max_w: Optional[int] = None):
        super().__init__()
        self.mode = mode
        self.init_k = float(init_k)
        self.max_h = max_h
        self.max_w = max_w
        self.k: Optional[nn.Parameter] = None
        self._C: Optional[int] = None

    def _maybe_init(self, x: torch.Tensor):
        H, W, C = _infer_hwc(x)
        if self.k is None:
            base = _param_base_shape(self.mode, H, W, C, max_h=self.max_h, max_w=self.max_w)
            self.k = nn.Parameter(torch.full(base, self.init_k, dtype=x.dtype, device=x.device))
            self._C = C
        else:
            if self.mode in ("dim", "voxel") and self._C is not None and C != self._C:
                raise ValueError(f"Channel size C changed from {self._C} to {C} for mode='{self.mode}'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._maybe_init(x)
        H, W, _ = _infer_hwc(x)
        k = self.k
        if k is None:
            return x

        if self.mode in ("token", "voxel"):
            k = _slice_hw(k, H, W)

        k = _broadcast_param_to_x(k, x)
        c = math.sqrt(2.0 / math.pi)
        kx = k * x
        u = c * (kx + 0.044715 * (kx ** 3))
        return 0.5 * x * (1.0 + torch.tanh(u))



# -------------------------
# straight-through helpers
# -------------------------
def _hard_sigmoid_st(p:torch.Tensor) -> torch.Tensor:
    soft = torch.sigmoid(p)
    hard = (soft >= 0.5).to(soft.dtype)
    return hard.detach() - soft.detach() + soft

def _hard_rezero_st(p: torch.Tensor) -> torch.Tensor:
    hard = (p > 0).to(p.dtype)
    return hard.detach() - p.detach() + p


class FlexFourier(nn.Module):
    kind = "fourier"

    def __init__(
        self,
        mode: FlexMode = "dim",
        n_terms: int = 4,
        init_scale: float = 0.01,
        max_h: Optional[int] = None,
        max_w: Optional[int] = None,
        use_gate: str = "none", # ["soft", "hard", "none", "residual"]
    ):
        super().__init__()
        self.mode = mode
        self.n_terms = int(n_terms)
        self.init_scale = float(init_scale)
        self.max_h = max_h
        self.max_w = max_w
        self.init_w = 1.0
        self.init_p = 0  # <-- FIX

        self.a: Optional[nn.Parameter] = None
        self.w: Optional[nn.Parameter] = None
        self.p: Optional[nn.Parameter] = None
        self._C: Optional[int] = None

        self.use_gate = use_gate
        self.init_t = 1

    def _maybe_init(self, x: torch.Tensor):
        H, W, C = _infer_hwc(x)
        if self.a is None:
            base = _param_base_shape(self.mode, H, W, C, max_h=self.max_h, max_w=self.max_w)
            shape = base + (self.n_terms,)

            # IMPORTANT: identity init => residual amplitude == 0
            a = torch.empty(shape, device=x.device, dtype=x.dtype).normal_(0.0, self.init_scale)

            w = torch.full(shape, self.init_w, device=x.device, dtype=x.dtype)
            p = torch.full(shape, self.init_p, device=x.device, dtype=x.dtype)

            self.a = nn.Parameter(a)
            self.w = nn.Parameter(w)
            self.p = nn.Parameter(p)
            self._C = C

            if self.use_gate != "none":
                # scale amplitudes to zero initially
                self.t = nn.Parameter(
                    torch.full((H, W, C), self.init_t, dtype=x.dtype, device=x.device)
                )

        else:
            if self.mode in ("dim", "voxel") and self._C is not None and C != self._C:
                raise ValueError(f"Channel size C changed from {self._C} to {C} for mode='{self.mode}'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._maybe_init(x)
        H, W, _ = _infer_hwc(x)

        a, w, p = self.a, self.w, self.p
        if a is None or w is None or p is None:
            return x

        if self.mode in ("token", "voxel"):
            a = _slice_hw(a, H, W)
            w = _slice_hw(w, H, W)
            p = _slice_hw(p, H, W)

        # x_e: [..., H, W, C, 1]
        x_e = x.unsqueeze(-1)

        # bring params to [..., H, W, C, T]
        while a.ndim < x_e.ndim:
            a = a.unsqueeze(0)
            w = w.unsqueeze(0)
            p = p.unsqueeze(0)

        residual = (a * torch.sin(w * x_e + p)).sum(dim=-1)  # [..., H, W, C]

        if self.use_gate == "none":
            return residual
        elif self.use_gate == "residual":
            return x + residual  # <-- identity at init when a==0
        else:
            p = self.t
            if self.use_gate == "soft":
                gate = torch.sigmoid(p)
            elif self.use_gate == "hard":
                gate = _hard_sigmoid_st(p)
            return residual * gate + x * (1 - gate)

class FlexSpline(nn.Module):
    kind = "spline"

    def __init__(
        self,
        mode: FlexMode,
        n_knots: int = 16,
        x_min: float = -3.0,
        x_max: float = 3.0,
        init: Literal["identity", "zero"] = "identity",
        max_h: Optional[int] = None,
        max_w: Optional[int] = None,
        use_gate: str = "none",  # ["soft", "hard", "none"]
    ):
        super().__init__()
        self.mode = mode
        self.n_knots = int(n_knots)
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.init = init
        self.max_h = max_h
        self.max_w = max_w
        self.init_eps = 1e-3

        self.register_buffer("knots_x", torch.linspace(self.x_min, self.x_max, steps=self.n_knots))
        self.knots_y: Optional[nn.Parameter] = None
        self._C: Optional[int] = None

        self.use_gate = use_gate
        self.init_t = 1

    def _maybe_init(self, x: torch.Tensor):
        H, W, C = _infer_hwc(x)

        if self.knots_x.device != x.device or self.knots_x.dtype != x.dtype:
            self.knots_x = self.knots_x.to(device=x.device, dtype=x.dtype)

        if self.knots_y is None:
            base = _param_base_shape(self.mode, H, W, C, max_h=self.max_h, max_w=self.max_w)
            shape = base + (self.n_knots,)

            # print("self.mode:", self.mode, "numel:", math.prod(base), "shape:", base)

            if self.init == "identity":
                ky = self.knots_x.view(1, 1, 1, -1).expand(*base, self.n_knots).clone()
                if self.init_eps > 0:
                    ky = ky + torch.empty_like(ky).normal_(0.0, self.init_eps)
            else:
                ky = torch.zeros(shape, dtype=x.dtype, device=x.device)

            self.knots_y = nn.Parameter(ky)
            self._C = C

            if self.use_gate != "none":
                # scale amplitudes to zero initially
                self.t = nn.Parameter(
                    torch.full((H, W, C), self.init_t, dtype=x.dtype, device=x.device)
                )


        else:
            if self.mode in ("dim", "voxel") and self._C is not None and C != self._C:
                raise ValueError(f"Channel size C changed from {self._C} to {C} for mode='{self.mode}'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._maybe_init(x)
        x_clamped = x.clamp(self.x_min, self.x_max)

        idx = torch.bucketize(x_clamped, self.knots_x) - 1
        idx = idx.clamp(0, self.n_knots - 2)

        x0 = self.knots_x[idx]
        x1 = self.knots_x[idx + 1]

        ky = self.knots_y
        if ky is None:
            return x

        # If you implemented slicing for spatial/voxel, do it BEFORE expand:
        H, W, C = _infer_hwc(x)
        if self.mode in ("token", "voxel"):
            ky = _slice_hw(ky, H, W)  # ky: [H,W,C,K] after slice

        # ky should now be [H',W',C',K]
        if ky.ndim != 4:
            raise ValueError(f"knots_y expected 4D [H,W,C,K], got {tuple(ky.shape)}")

        # Make ky [1,H,W,C,K] then expand to [B,H,W,C,K]
        while ky.ndim < x.ndim + 1:
            ky = ky.unsqueeze(0)

        B = x.shape[0]
        ky = ky.expand(B, H, W, C, self.n_knots)

        y0 = torch.gather(ky, dim=-1, index=idx.unsqueeze(-1)).squeeze(-1)
        y1 = torch.gather(ky, dim=-1, index=(idx + 1).unsqueeze(-1)).squeeze(-1)

        t = (x_clamped - x0) / (x1 - x0 + 1e-12)
        residual = t * (y1 - y0)

        if self.use_gate == "none":
            # spline value = y0 + t*(y1 - y0); the bare increment omitted y0 (bug).
            return y0 + residual
        elif self.use_gate == "residual":
            return y0 + residual  # <-- identity at init when a==0
        else:
            p = self.t
            if self.use_gate == "soft":
                gate = torch.sigmoid(p)
            elif self.use_gate == "hard":
                gate = _hard_sigmoid_st(p)
            # print("residual,gate, y0 shapes:", residual.shape, gate.shape, y0.shape, p.shape)
            return residual * gate + y0 * (1 - gate)



        # return y0 + t * (y1 - y0)


class FlexPolynomial(nn.Module):
    kind = "polynomial"

    def __init__(
        self,
        mode: FlexMode,
        degree: int = 3,
        init: Literal["identity", "zero"] = "identity",
        max_h: Optional[int] = None,
        max_w: Optional[int] = None,
        use_gate: str = "none",  # ["soft", "hard", "none"]
    ):
        super().__init__()
        self.mode = mode
        self.degree = int(degree)
        self.init = init
        self.max_h = max_h
        self.max_w = max_w
        self.init_scale = 1e-3


        self.c: Optional[nn.Parameter] = None
        self._C: Optional[int] = None

        self.use_gate = use_gate
        self.init_t = 1

    def _maybe_init(self, x: torch.Tensor):
        H, W, C = _infer_hwc(x)
        if self.c is None:
            base = _param_base_shape(self.mode, H, W, C, max_h=self.max_h, max_w=self.max_w)
            shape = base + (self.degree + 1,)

            c = torch.zeros(shape, dtype=x.dtype, device=x.device)

            if self.init == "identity":
                if self.degree >= 1:
                    c[..., 1] = 1.0
                # tiny higher-order terms
                if self.init_scale > 0 and self.degree >= 2:
                    c[..., 2:] = torch.empty_like(c[..., 2:]).normal_(0.0, self.init_scale)
            else:
                if self.init_scale > 0:
                    c = c + torch.empty_like(c).normal_(0.0, self.init_scale)

            self.c = nn.Parameter(c)
            self._C = C

            if self.use_gate != "none":
                # scale amplitudes to zero initially
                self.t = nn.Parameter(
                    torch.full((H, W, C), self.init_t, dtype=x.dtype, device=x.device)
                )


        else:
            if self.mode in ("dim", "voxel") and self._C is not None and C != self._C:
                raise ValueError(f"Channel size C changed from {self._C} to {C} for mode='{self.mode}'.")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._maybe_init(x)
        H, W, _ = _infer_hwc(x)

        c = self.c
        if c is None:
            return x

        if self.mode in ("token", "voxel"):
            c = _slice_hw(c, H, W)

        # broadcast to x with coeff dim
        while c.ndim < x.ndim + 1:
            c = c.unsqueeze(0)

        # Horner
        y = c[..., -1]
        for k in range(self.degree - 1, -1, -1):
            y = y * x + c[..., k]

        residual = y

        if self.use_gate == "none":
            return residual
        elif self.use_gate == "residual":
            return x + residual  # <-- identity at init when a==0
        else:
            p = self.t
            if self.use_gate == "soft":
                gate = torch.sigmoid(p)
            elif self.use_gate == "hard":
                gate = _hard_sigmoid_st(p)

            return residual * gate + x * (1 - gate)

        # return y


# -----------------------
# Factory
# -----------------------

class FlexTanhRes(nn.Module):
    """phi(z) = z + a * tanh(b * z), with per-channel a (init 0) and b (init 1).

    Designed against what the gradient probe actually measured on a 7B run:

    * The spline's knots are updated by bucketed interpolation, so each knot only
      sees the samples that land in its bin -- its gradient came out ~350x smaller
      than B's and the "learned" nonlinearity barely moved. Here every sample
      contributes to every parameter (d/da = tanh(bz), d/db = a*z*sech^2(bz)), so
      the signal is dense.
    * The spline needs its input inside a fixed knot range, which forced a
      pre-activation LayerNorm -- and that norm was what pushed the initial function
      away from LoRA, inflated B's gradient into the clipping regime, and erased the
      per-token magnitude the input-conditional claim rests on. tanh is bounded for
      any input scale, so no pre-normalization is needed at all.
    * a = 0 at init makes phi EXACTLY the identity, i.e. an exact LoRA starting
      point that does not depend on the gate being initialized closed.

    Costs 2r parameters per module instead of the spline's 16r.
    """

    kind = "tanhres"

    def __init__(self, mode: FlexMode, a_init: float = 0.0, b_init: float = 1.0,
                 max_h: Optional[int] = None, max_w: Optional[int] = None,
                 use_gate: str = "none"):
        super().__init__()
        self.mode = mode
        self.a_init = float(a_init)
        self.b_init = float(b_init)
        self.max_h = max_h
        self.max_w = max_w
        self.use_gate = use_gate
        self.a: Optional[nn.Parameter] = None
        self.b: Optional[nn.Parameter] = None

    def _maybe_init(self, x: torch.Tensor):
        if self.a is not None:
            return
        H, W, C = _infer_hwc(x)
        shape = _param_base_shape(self.mode, H, W, C, max_h=self.max_h, max_w=self.max_w)
        self.a = nn.Parameter(torch.full(shape, self.a_init, dtype=x.dtype, device=x.device))
        self.b = nn.Parameter(torch.full(shape, self.b_init, dtype=x.dtype, device=x.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._maybe_init(x)
        return x + self.a * torch.tanh(self.b * x)


class FlexRankMix(nn.Module):
    """h = z * (1 + a * tanh(W z + b)) -- conditional mixing ACROSS ranks.

    Every nonlinear-LoRA method we compete with (AuroRA, CeRA, AFA-LoRA, LoRAN, and
    the activation-adaptation line) applies phi ELEMENTWISE. An elementwise phi can
    reshape each coordinate of the code independently, but it can never change the
    relative weighting of the rank-1 components of B A, because phi(z)_i depends only
    on z_i. The "input-conditional mixture of low-rank updates" the method claims is
    therefore out of reach of an elementwise nonlinearity by construction -- which is
    consistent with this project's own CPU probe, where a bilinear adapter reached
    family_rank 10.5 against the spline's 4.

    Here the scaling applied to rank i is a function of ALL ranks, so each token gets
    a different combination of the rank-1 directions: literally a mixture of LoRAs.

    a is initialized to 0, so phi is EXACTLY the identity at init and the adapter
    starts as exact LoRA. Costs r^2 + 2r parameters per module (24 at r=4).
    """

    kind = "rankmix"

    def __init__(self, mode: FlexMode, a_init: float = 0.0, w_init: Optional[float] = None,
                 max_h: Optional[int] = None, max_w: Optional[int] = None,
                 use_gate: str = "none"):
        super().__init__()
        self.mode = mode
        self.a_init = float(a_init)
        self.w_init = None if w_init is None else float(w_init)
        self.use_gate = use_gate
        self.a: Optional[nn.Parameter] = None
        self.W: Optional[nn.Parameter] = None
        self.b: Optional[nn.Parameter] = None

    def _maybe_init(self, x: torch.Tensor):
        if self.a is not None:
            return
        C = int(x.shape[-1])
        self.a = nn.Parameter(torch.full((C,), self.a_init, dtype=x.dtype, device=x.device))
        # W must start at a normal layer scale, NOT small. a is 0 at init (that is what
        # buys the exact LoRA starting point), and dL/dW is proportional to a, so W's
        # gradient is exactly 0 on the first step and W can only move once a does.
        # Meanwhile dL/da is proportional to tanh(Wz): a small W makes that ~0.07 and a
        # crawls, which deadlocks both. Measured at r=4: |dL/da| = 0.093 at w_init=0.02
        # versus 0.317 at 1/sqrt(r). With w_init=0.02 the amplitude reached only
        # |a| ~ 0.0035 after 125 steps and did not react to task conflict at all.
        w = self.w_init if self.w_init is not None else C ** -0.5
        self.W = nn.Parameter(torch.randn(C, C, dtype=x.dtype, device=x.device) * w)
        self.b = nn.Parameter(torch.zeros(C, dtype=x.dtype, device=x.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._maybe_init(x)
        mix = torch.tanh(torch.matmul(x, self.W.transpose(0, 1)) + self.b)
        return x * (1.0 + self.a * mix)


class FlexRankMix2(nn.Module):
    """h = z + a * MLP(z): a 2-layer cross-rank MLP, exact identity at init (a=0).

    rankmix uses a single tanh(Wz); this pushes the cross-rank interaction to a
    proper 2-layer MLP (r -> hidden -> r, GELU) to find the ceiling of "how much does
    mixing across ranks actually buy". Still exact LoRA at init because the output is
    scaled by a=0. Costs r*h + h + h*r + r + r params (~ r=8,h=32: 552) per module.
    """
    kind = "rankmix2"

    def __init__(self, mode: FlexMode, a_init: float = 0.0, hidden_mult: int = 4,
                 max_h=None, max_w=None, use_gate: str = "none"):
        super().__init__()
        self.mode = mode; self.a_init = float(a_init); self.hidden_mult = int(hidden_mult)
        self.a = None; self.w1 = None; self.b1 = None; self.w2 = None; self.b2 = None

    def _maybe_init(self, x):
        if self.a is not None: return
        C = int(x.shape[-1]); H = C * self.hidden_mult
        d, dev = x.dtype, x.device
        self.w1 = nn.Parameter(torch.randn(H, C, dtype=d, device=dev) * C ** -0.5)
        self.b1 = nn.Parameter(torch.zeros(H, dtype=d, device=dev))
        self.w2 = nn.Parameter(torch.randn(C, H, dtype=d, device=dev) * H ** -0.5)
        self.b2 = nn.Parameter(torch.zeros(C, dtype=d, device=dev))
        self.a = nn.Parameter(torch.full((C,), self.a_init, dtype=d, device=dev))

    def forward(self, x):
        self._maybe_init(x)
        h = torch.nn.functional.gelu(torch.matmul(x, self.w1.transpose(0, 1)) + self.b1)
        h = torch.matmul(h, self.w2.transpose(0, 1)) + self.b2
        return x + self.a * h


class FlexBilinear(nn.Module):
    """h = z + a * ((Uz) * (Vz)): explicit second-order z_i z_j interaction.

    The probe in this project measured a bilinear adapter reaching family_rank 10.5
    against the spline's 4, but the LLM code only ever shipped elementwise activations.
    (Uz)*(Vz) is a rank-controlled bilinear form: it produces genuine cross-rank
    products z_i z_j, unlike any elementwise phi. a=0 keeps the exact LoRA start.
    Costs 2*k*r + r params (k = bilinear rank).
    """
    kind = "bilinear"

    def __init__(self, mode: FlexMode, a_init: float = 0.0, k_mult: int = 2,
                 max_h=None, max_w=None, use_gate: str = "none"):
        super().__init__()
        self.mode = mode; self.a_init = float(a_init); self.k_mult = int(k_mult)
        self.a = None; self.U = None; self.V = None

    def _maybe_init(self, x):
        if self.a is not None: return
        C = int(x.shape[-1]); K = C * self.k_mult
        d, dev = x.dtype, x.device
        self.U = nn.Parameter(torch.randn(K, C, dtype=d, device=dev) * C ** -0.5)
        self.V = nn.Parameter(torch.randn(C, K, dtype=d, device=dev) * K ** -0.5)
        self.a = nn.Parameter(torch.full((C,), self.a_init, dtype=d, device=dev))

    def forward(self, x):
        self._maybe_init(x)
        u = torch.matmul(x, self.U.transpose(0, 1))
        h = torch.matmul(u * u, self.V.transpose(0, 1))   # (Uz).(Uz) folded through V
        return x + self.a * h


class CompAFALoRA(nn.Module):
    """AFA-LoRA (arXiv 2512.22455): phi(z)=beta*relu(z)+(1-beta)*z, beta ANNEALED 1->0.
    No extra learnable params; nonlinearity anneals to linear over first 30% of steps.
    We expose beta as a buffer set by a scheduler callback; default fixed for ablation."""
    kind = "afa"
    def __init__(self, mode="dim", **kw):
        super().__init__(); self.register_buffer("beta", torch.tensor(1.0))
    def forward(self, x):
        return self.beta*torch.relu(x) + (1.0-self.beta)*x


class AuroRAG(nn.Module):
    """AuroRA + provable LoRA fallback: phi(z) = z + a * [tanh(H tanh(z)) + w_s*spline(z)].

    AuroRA's sigma has NO clean linear fallback (nested tanh can't be identity). We wrap
    its ANL in a residual scaled by a per-dim amplitude a initialized to 0, so at init
    phi(z)=z EXACTLY (=> exact LoRA start, provable fallback), and the model learns how
    much AuroRA-style nonlinearity to add. The layer-level input gate (gate_mode=input)
    composes on top, giving AuroRA the input-conditioning it lacks.
    """
    kind = "aurorag"

    def __init__(self, mode="dim", n_knots=8, a_init=0.0, **kw):
        super().__init__(); self.n_knots=int(n_knots); self.a_init=float(a_init)
        self.H=None; self.ws=None; self.ky=None; self.a=None
        self.register_buffer("kx", torch.linspace(-3,3,self.n_knots))
    def _init(self,x):
        if self.H is not None: return
        C=int(x.shape[-1]); d,dev=x.dtype,x.device
        self.H=nn.Parameter(torch.eye(C,dtype=d,device=dev)+0.01*torch.randn(C,C,dtype=d,device=dev))
        self.ws=nn.Parameter(torch.zeros(C,dtype=d,device=dev))
        self.ky=nn.Parameter(self.kx.to(d).view(1,-1).repeat(C,1).clone())
        self.a=nn.Parameter(torch.full((C,), self.a_init, dtype=d, device=dev))
    def forward(self,x):
        self._init(x)
        fixed=torch.tanh(torch.matmul(torch.tanh(x), self.H.transpose(0,1)))
        kx=self.kx.to(x.dtype); idx=torch.searchsorted(kx, x.clamp(kx[0],kx[-1]).contiguous())
        idx=idx.clamp(1,self.n_knots-1)
        yv=self.ky[torch.arange(x.shape[-1],device=x.device), idx]
        return x + self.a * (fixed + self.ws*yv)


class AuroRAF(nn.Module):
    """AuroRA with provable LoRA fallback via per-dim interpolation gate.
    phi(z)=(1-g)*T(z)+g*z, g=sigmoid(p), p init -6 => g~0 => phi~AuroRA at init;
    p->+inf => g->1 => phi=z exact LoRA fallback. No spurious residual (cf aurorag)."""
    kind = "auroraf"

    def __init__(self, mode="dim", n_knots=8, p_init=-6.0, **kw):
        super().__init__(); self.n_knots=int(n_knots); self.p_init=float(p_init)
        self.H=None; self.ws=None; self.ky=None; self.p=None
        self.register_buffer("kx", torch.linspace(-3,3,self.n_knots))
    def _init(self,x):
        if self.H is not None: return
        C=int(x.shape[-1]); d,dev=x.dtype,x.device
        self.H=nn.Parameter(torch.eye(C,dtype=d,device=dev)+0.01*torch.randn(C,C,dtype=d,device=dev))
        self.ws=nn.Parameter(torch.zeros(C,dtype=d,device=dev))
        self.ky=nn.Parameter(self.kx.to(d).view(1,-1).repeat(C,1).clone())
        self.p=nn.Parameter(torch.full((C,), self.p_init, dtype=d, device=dev))
    def forward(self,x):
        self._init(x)
        fixed=torch.tanh(torch.matmul(torch.tanh(x), self.H.transpose(0,1)))
        kx=self.kx.to(x.dtype); idx=torch.searchsorted(kx, x.clamp(kx[0],kx[-1]).contiguous())
        idx=idx.clamp(1,self.n_knots-1)
        yv=self.ky[torch.arange(x.shape[-1],device=x.device), idx]
        T=fixed+self.ws*yv
        g=torch.sigmoid(self.p)
        return (1.0-g)*T + g*x


class FlexGLU(nn.Module):
    """GLU-style MULTIPLICATIVE bottleneck: glu(z) = (W1 z) * sigmoid(W2 z + b).
    Multiplicative (bilinear) nonlinearity in the code, vs AuroRA's additive tanh/spline.
    Init W1=I, W2=0, b=+3 => gate~0.95 => glu ~= z (near-LoRA start)."""
    kind = "glu"
    def __init__(self, mode="dim", **kw):
        super().__init__(); self.W1=None; self.W2=None; self.b=None
    def _init(self, x):
        if self.W1 is not None: return
        C=int(x.shape[-1]); d,dev=x.dtype,x.device
        self.W1=nn.Parameter(torch.eye(C,dtype=d,device=dev))
        self.W2=nn.Parameter(torch.zeros(C,C,dtype=d,device=dev))
        self.b =nn.Parameter(torch.full((C,),3.0,dtype=d,device=dev))
    def forward(self, x):
        self._init(x)
        return torch.matmul(x, self.W1.transpose(0,1)) * torch.sigmoid(torch.matmul(x, self.W2.transpose(0,1)) + self.b)


class AuroGLU(nn.Module):
    """AuroRA transform gated by an input-dependent GLU sigmoid:
    sigma(z) = [tanh(H tanh z) + w_s*spline(z)] * sigmoid(W2 z + b).
    Combines additive (AuroRA) + multiplicative (GLU) nonlinearity in the bottleneck."""
    kind = "auroglu"
    def __init__(self, mode="dim", n_knots=8, **kw):
        super().__init__(); self.n_knots=int(n_knots)
        self.H=None; self.ws=None; self.ky=None; self.W2=None; self.b=None
        self.register_buffer("kx", torch.linspace(-3,3,self.n_knots))
    def _init(self,x):
        if self.H is not None: return
        C=int(x.shape[-1]); d,dev=x.dtype,x.device
        self.H=nn.Parameter(torch.eye(C,dtype=d,device=dev)+0.01*torch.randn(C,C,dtype=d,device=dev))
        self.ws=nn.Parameter(torch.zeros(C,dtype=d,device=dev))
        self.ky=nn.Parameter(self.kx.to(d).view(1,-1).repeat(C,1).clone())
        self.W2=nn.Parameter(torch.zeros(C,C,dtype=d,device=dev))
        self.b =nn.Parameter(torch.full((C,),3.0,dtype=d,device=dev))
    def forward(self,x):
        self._init(x)
        fixed=torch.tanh(torch.matmul(torch.tanh(x), self.H.transpose(0,1)))
        kx=self.kx.to(x.dtype); idx=torch.searchsorted(kx, x.clamp(kx[0],kx[-1]).contiguous()).clamp(1,self.n_knots-1)
        yv=self.ky[torch.arange(x.shape[-1],device=x.device), idx]
        aur=fixed+self.ws*yv
        gate=torch.sigmoid(torch.matmul(x,self.W2.transpose(0,1))+self.b)
        return aur*gate


class CompAuroRA(nn.Module):
    """AuroRA (NeurIPS'25, 2505.18738): sigma(Z)=tanh(H tanh(Z)) + w_s*spline(Z).
    H is r~xr~ (cross-rank!), w_s per-dim spline weights. No exact LoRA fallback."""
    kind = "aurora"
    def __init__(self, mode="dim", n_knots=8, **kw):
        super().__init__(); self.n_knots=int(n_knots); self.H=None; self.ws=None; self.ky=None
        self.register_buffer("kx", torch.linspace(-3,3,self.n_knots))
    def _init(self,x):
        if self.H is not None: return
        C=int(x.shape[-1]); d,dev=x.dtype,x.device
        self.H=nn.Parameter(torch.eye(C,dtype=d,device=dev)+0.01*torch.randn(C,C,dtype=d,device=dev))
        self.ws=nn.Parameter(torch.zeros(C,dtype=d,device=dev))
        self.ky=nn.Parameter(self.kx.to(d).view(1,-1).repeat(C,1).clone())
    def forward(self,x):
        self._init(x)
        fixed=torch.tanh(torch.matmul(torch.tanh(x), self.H.transpose(0,1)))
        kx=self.kx.to(x.dtype); idx=torch.searchsorted(kx, x.clamp(kx[0],kx[-1]).contiguous())
        idx=idx.clamp(1,self.n_knots-1)
        yv=self.ky[torch.arange(x.shape[-1],device=x.device), idx]  # crude per-dim spline lookup
        return fixed + self.ws*yv


class CompLoRAN(nn.Module):
    """LoRAN (EMNLP'24 Findings): Sinter(x)=A*sin(w*x)*x + x, A=5e-5, w=1e4 FIXED (not learned).
    Applied elementwise. A=0 recovers LoRA."""
    kind = "loran"
    def __init__(self, mode="dim", amp=5e-5, freq=1e4, **kw):
        super().__init__(); self.amp=float(amp); self.freq=float(freq)
    def forward(self,x):
        return self.amp*torch.sin(self.freq*x)*x + x


class FlexRankMixC(nn.Module):
    """Compressed-bottleneck cross-rank nonlinearity, AuroRA's key idea inside LeNA.

    AuroRA beats LeNA on GSM8K (0.287 vs 0.224). The hypothesis: its win comes from
    compressing the code to r~ << r BEFORE the nonlinearity, so the bottleneck truly
    binds and the nonlinearity becomes necessary -- consistent with this project's
    finding that nonlinearity only helps when rank is the constraint. LeNA at r=8 does
    not bind hard enough (a -> 0). Here we down-project z (dim r) to r~ = max(2, r//k),
    apply a (nested-tanh) cross-rank map there, and up-project back, added as residual.

      u = Wd z            (r -> r~)          # compress
      m = tanh(H tanh(u)) (r~ -> r~)         # AuroRA-style nested cross-rank
      out = z + a * (Wu m) (r~ -> r)         # expand, residual; a=0 -> exact identity
    """
    kind = "rankmixc"

    def __init__(self, mode: FlexMode, a_init: float = 0.0, compress: int = 4,
                 max_h=None, max_w=None, use_gate: str = "none"):
        super().__init__()
        self.mode = mode; self.a_init = float(a_init); self.compress = int(compress)
        self.a = None; self.Wd = None; self.H = None; self.Wu = None

    def _maybe_init(self, x):
        if self.a is not None: return
        C = int(x.shape[-1]); rt = max(2, C // self.compress)
        d, dev = x.dtype, x.device
        self.Wd = nn.Parameter(torch.randn(rt, C, dtype=d, device=dev) * C ** -0.5)
        self.H  = nn.Parameter(torch.eye(rt, dtype=d, device=dev) + 0.01*torch.randn(rt, rt, dtype=d, device=dev))
        self.Wu = nn.Parameter(torch.randn(C, rt, dtype=d, device=dev) * rt ** -0.5)
        self.a  = nn.Parameter(torch.full((C,), self.a_init, dtype=d, device=dev))

    def forward(self, x):
        self._maybe_init(x)
        u = torch.matmul(x, self.Wd.transpose(0, 1))          # compress r->r~
        m = torch.tanh(torch.matmul(torch.tanh(u), self.H.transpose(0, 1)))  # nested cross-rank
        return x + self.a * torch.matmul(m, self.Wu.transpose(0, 1))         # expand + residual

def make_lena_activation(kind: ActKind, mode: FlexMode, **kwargs: Any) -> nn.Module:
    k = str(kind).lower()
    if k == "identity":
        act = IdentityAct()
    elif k == "relu":
        act = FlexReLU(mode=mode, **kwargs)
    elif k == "gelu":
        act = FlexGELU(mode=mode, **kwargs)
    elif k == "swish":
        act = FlexSwish(mode=mode, **kwargs)
    elif k == "fourier":
        act = FlexFourier(mode=mode, **kwargs)
    elif k == "spline":
        act = FlexSpline(mode=mode, **kwargs)
    elif k == "polynomial":
        act = FlexPolynomial(mode=mode, **kwargs)
    elif k == "tanhres":
        kwargs.pop("use_gate", None)
        act = FlexTanhRes(mode=mode, **kwargs)
    elif k == "rankmix":
        kwargs.pop("use_gate", None)
        act = FlexRankMix(mode=mode, **kwargs)
    elif k == "rankmix2":
        kwargs.pop("use_gate", None)
        act = FlexRankMix2(mode=mode, **kwargs)
    elif k == "bilinear":
        kwargs.pop("use_gate", None)
        act = FlexBilinear(mode=mode, **kwargs)
    elif k == "afa":
        act = CompAFALoRA(mode=mode, **kwargs)
    elif k == "auroglu":
        kwargs.pop("use_gate", None)
        act = AuroGLU(mode=mode, **kwargs)
    elif k == "glu":
        kwargs.pop("use_gate", None)
        act = FlexGLU(mode=mode, **kwargs)
    elif k == "aurora":
        act = CompAuroRA(mode=mode, **kwargs)
    elif k == "loran":
        act = CompLoRAN(mode=mode, **kwargs)
    elif k == "rankmixc":
        kwargs.pop("use_gate", None)
        act = FlexRankMixC(mode=mode, **kwargs)
    elif k == "auroraf":
        kwargs.pop("use_gate", None)
        act = AuroRAF(mode=mode, **kwargs)
    elif k == "aurorag":
        kwargs.pop("use_gate", None)
        act = AuroRAG(mode=mode, **kwargs)
    else:
        raise ValueError(f"Unknown lena activation kind: {kind}")

    # helpful for debugging / FLOPs estimation
    setattr(act, "kind", k)
    return act
