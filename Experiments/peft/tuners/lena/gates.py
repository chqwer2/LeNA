import torch
import torch.nn as nn
from typing import Optional, Literal

GateType = Literal["none", "sigmoid", "rezero"]
GateMode = Literal["global", "spatial", "channel", "voxel"]
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
    ):
        super().__init__()
        self.gate_type = gate_type
        self.gate_mode = gate_mode
        self.gate_strength = gate_strength
        self.init = 1 # float(init)
        self.dtype = dtype
        self.device = device

        # lazily initialized
        self.param: Optional[nn.Parameter] = None

        if gate_type == "none":
            return

    # -------------------------
    # lazy initialization
    # -------------------------
    def _init_param_from_x(self, x: torch.Tensor):
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
