from __future__ import annotations

from typing import Any, Literal, Optional, Tuple

import math
import torch
import torch.nn as nn

FlexMode = Literal["global", "spatial", "channel", "voxel"]
ActKind = Literal["identity", "relu", "gelu", "fourier", "spline", "polynomial"]


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
    if mode in ("spatial", "voxel"):
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
    if mode == "channel":
        return (1, 1, C)
    if mode == "spatial":
        max_h, max_w = _require_max_hw(mode, max_h, max_w)
        return (int(max_h), int(max_w), 1)
    if mode == "voxel":
        max_h, max_w = _require_max_hw(mode, max_h, max_w)
        return (int(max_h), int(max_w), C)
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
        self._maybe_init(x)
        return torch.where(x >= 0, x, 0)


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
            if self.mode in ("channel", "voxel") and self._C is not None and C != self._C:
                raise ValueError(f"Channel size C changed from {self._C} to {C} for mode='{self.mode}'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._maybe_init(x)
        H, W, _ = _infer_hwc(x)
        k = self.k
        if k is None:
            return x

        if self.mode in ("spatial", "voxel"):
            k = _slice_hw(k, H, W)

        k = _broadcast_param_to_x(k, x)
        c = math.sqrt(2.0 / math.pi)
        kx = k * x
        u = c * (kx + 0.044715 * (kx ** 3))
        return 0.5 * x * (1.0 + torch.tanh(u))


class FlexFourier(nn.Module):
    kind = "fourier"

    def __init__(
        self,
        mode: FlexMode = "channel",
        n_terms: int = 4,
        init_scale: float = 0.01,
        max_h: Optional[int] = None,
        max_w: Optional[int] = None,
    ):
        super().__init__()
        self.mode = mode
        self.n_terms = int(n_terms)
        self.init_scale = float(init_scale)
        self.max_h = max_h
        self.max_w = max_w

        self.a: Optional[nn.Parameter] = None
        self.w: Optional[nn.Parameter] = None
        self.p: Optional[nn.Parameter] = None
        self._C: Optional[int] = None

    def _maybe_init(self, x: torch.Tensor):
        H, W, C = _infer_hwc(x)
        if self.a is None:
            base = _param_base_shape(self.mode, H, W, C, max_h=self.max_h, max_w=self.max_w)
            shape = base + (self.n_terms,)

            # residual amplitude tiny => near-identity
            a = torch.empty(shape, device=x.device, dtype=x.dtype).normal_(0.0, self.init_scale)
            w = torch.full(shape, self.init_w, device=x.device, dtype=x.dtype)
            p = torch.full(shape, self.init_p, device=x.device, dtype=x.dtype)

            self.a = nn.Parameter(a)
            self.w = nn.Parameter(w)
            self.p = nn.Parameter(p)
            self._C = C
        else:
            if self.mode in ("channel", "voxel") and self._C is not None and C != self._C:
                raise ValueError(f"Channel size C changed from {self._C} to {C} for mode='{self.mode}'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._maybe_init(x)
        H, W, _ = _infer_hwc(x)

        a, w, p = self.a, self.w, self.p
        if a is None or w is None or p is None:
            return x

        # slice spatial tables if needed
        if self.mode in ("spatial", "voxel"):
            a = _slice_hw(a, H, W)
            w = _slice_hw(w, H, W)
            p = _slice_hw(p, H, W)

        # broadcast to x with extra term dim
        # x_e: [..., H, W, C, 1]
        x_e = x.unsqueeze(-1)

        # bring params to [..., H, W, C, T]
        while a.ndim < x_e.ndim:
            a = a.unsqueeze(0)
            w = w.unsqueeze(0)
            p = p.unsqueeze(0)

        y = (a * torch.sin(w * x_e + p)).sum(dim=-1)  # [..., H, W, C]
        return y


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
    ):
        super().__init__()
        self.mode = mode
        self.n_knots = int(n_knots)
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.init = init
        self.max_h = max_h
        self.max_w = max_w

        self.register_buffer("knots_x", torch.linspace(self.x_min, self.x_max, steps=self.n_knots))
        self.knots_y: Optional[nn.Parameter] = None
        self._C: Optional[int] = None

    def _maybe_init(self, x: torch.Tensor):
        H, W, C = _infer_hwc(x)
        if self.knots_x.device != x.device or self.knots_x.dtype != x.dtype:
            self.knots_x = self.knots_x.to(device=x.device, dtype=x.dtype)

        if self.knots_y is None:
            base = _param_base_shape(self.mode, H, W, C, max_h=self.max_h, max_w=self.max_w)
            shape = base + (self.n_knots,)

            if self.init == "identity":
                ky = self.knots_x.view(1, 1, 1, -1).expand(*base, self.n_knots).clone()
                if self.init_eps > 0:
                    ky = ky + torch.empty_like(ky).normal_(0.0, self.init_eps)
            else:
                ky = torch.zeros(shape, dtype=x.dtype, device=x.device)

            self.knots_y = nn.Parameter(ky)
            self._C = C
        else:
            if self.mode in ("channel", "voxel") and self._C is not None and C != self._C:
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
        if self.mode in ("spatial", "voxel"):
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
        return y0 + t * (y1 - y0)


class FlexPolynomial(nn.Module):
    kind = "polynomial"

    def __init__(
        self,
        mode: FlexMode,
        degree: int = 3,
        init: Literal["identity", "zero"] = "identity",
        max_h: Optional[int] = None,
        max_w: Optional[int] = None,
    ):
        super().__init__()
        self.mode = mode
        self.degree = int(degree)
        self.init = init
        self.max_h = max_h
        self.max_w = max_w

        self.c: Optional[nn.Parameter] = None
        self._C: Optional[int] = None

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
        else:
            if self.mode in ("channel", "voxel") and self._C is not None and C != self._C:
                raise ValueError(f"Channel size C changed from {self._C} to {C} for mode='{self.mode}'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._maybe_init(x)
        H, W, _ = _infer_hwc(x)

        c = self.c
        if c is None:
            return x

        if self.mode in ("spatial", "voxel"):
            c = _slice_hw(c, H, W)

        # broadcast to x with coeff dim
        while c.ndim < x.ndim + 1:
            c = c.unsqueeze(0)

        # Horner
        y = c[..., -1]
        for k in range(self.degree - 1, -1, -1):
            y = y * x + c[..., k]
        return y


# -----------------------
# Factory
# -----------------------

def make_flora_activation(kind: ActKind, mode: FlexMode, **kwargs: Any) -> nn.Module:
    k = str(kind).lower()
    if k == "identity":
        act = IdentityAct()
    elif k == "relu":
        act = FlexReLU(mode=mode, **kwargs)
    elif k == "gelu":
        act = FlexGELU(mode=mode, **kwargs)
    elif k == "fourier":
        act = FlexFourier(mode=mode, **kwargs)
    elif k == "spline":
        act = FlexSpline(mode=mode, **kwargs)
    elif k == "polynomial":
        act = FlexPolynomial(mode=mode, **kwargs)
    else:
        raise ValueError(f"Unknown flora activation kind: {kind}")

    # helpful for debugging / FLOPs estimation
    setattr(act, "kind", k)
    return act
