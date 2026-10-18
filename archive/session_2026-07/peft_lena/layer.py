from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
from .activations import make_lena_activation
from .config import LeNAConfig
from .gates import Gate

from copy import deepcopy

import torch
import torch.nn.functional as F
from torch import nn

from peft.utils.integrations import dequantize_module_weight, gather_params_ctx
from peft.utils.other import transpose


class SharedRMS(nn.Module):
    """Divide the code by ONE running scalar, shared over tokens and channels.

    The point of normalizing before phi is only to land the code inside the spline's
    knot range. A per-token LayerNorm does that but also forces every token's code to
    the same norm, which (a) moves the initial function far away from LoRA even when
    phi is identity-initialized, (b) inflates B's gradient enough to trip global grad
    clipping every step -- which then scales *everything* down and starves the
    activation and gate of updates -- and (c) deletes the per-token magnitude that the
    input-conditional story depends on. A single shared scalar fixes the range without
    touching the relative magnitudes.
    """

    def __init__(self, momentum: float = 0.01, eps: float = 1e-5, init: float = 1.0):
        super().__init__()
        self.momentum = float(momentum)
        self.eps = float(eps)
        self.register_buffer("rms", torch.tensor(float(init)))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if self.training:
            cur = z.detach().float().pow(2).mean().sqrt().clamp_min(self.eps)
            self.rms.mul_(1.0 - self.momentum).add_(self.momentum * cur)
        return z / (self.rms.to(z.dtype) + self.eps)


def _to_hwc(z: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int], int]:
    orig_ndim = z.ndim
    if z.ndim >= 4:
        H, W, C = int(z.shape[-3]), int(z.shape[-2]), int(z.shape[-1])
        return z, (H, W, C), orig_ndim
    if z.ndim == 3:
        H, C = int(z.shape[-2]), int(z.shape[-1])
        return z.unsqueeze(-2), (H, 1, C), orig_ndim
    if z.ndim == 2:
        C = int(z.shape[-1])
        return z.unsqueeze(-2).unsqueeze(-2), (1, 1, C), orig_ndim
    raise ValueError(f"Unsupported z shape: {tuple(z.shape)}")


def _from_hwc(z_hwc: torch.Tensor, orig_ndim: int) -> torch.Tensor:
    if orig_ndim >= 4:
        return z_hwc
    if orig_ndim == 3:
        return z_hwc.squeeze(-2)
    if orig_ndim == 2:
        return z_hwc.squeeze(-2).squeeze(-2)
    raise ValueError("orig_ndim must be >=2")


class SteerHead(nn.Module):
    """Input-steered rank-k correction that ESCAPES col(B).
    delta(z) = sum_j s_j(z) * (U @ softmax(Wr_j z)), s_j(z)=<ws_j,z>, ws init 0
    => starts at EXACT LoRA. The write direction U*softmax(Wr z) depends on the input
    code z, so the update leaves the fixed subspace col(B) -- the axis LoRA/AuroRA/CeRA
    cannot reach (their output stays in col(B) for any bottleneck nonlinearity)."""
    def __init__(self, d_out, r, p=4, k=1, init_U=None):
        super().__init__()
        self.k=int(k)
        if init_U is not None:
            U0=torch.zeros(d_out,int(p))
            c=min(int(p), init_U.shape[1]); U0[:, :c]=init_U[:, :c].float()
            self.U=nn.Parameter(U0)                      # SVD-init: weight-aligned directions
        else:
            self.U=nn.Parameter(torch.randn(d_out,int(p))/(float(d_out)**0.5))
        self.Wr=nn.Parameter(torch.randn(self.k,int(p),int(r))/(float(r)**0.5))
        self.ws=nn.Parameter(torch.zeros(self.k,int(r)))
        self.scale=1.0/float(int(r)**0.5)
    def forward(self,z):
        zc=z.to(self.U.dtype)                                   # [...,r]
        out=0.0
        for j in range(self.k):
            rho=torch.softmax(zc@self.Wr[j].t(),dim=-1)         # [...,p]
            sj=torch.tanh((zc*self.ws[j]).sum(-1,keepdim=True)) # [...,1] BOUNDED amplitude
            u=rho@self.U.t()                                    # [...,d_out] direction
            u=u/(u.norm(dim=-1,keepdim=True)+1e-6)              # UNIT direction (no blow-up)
            out=out+sj*u
        return out * self.scale


class LeNALinear(nn.Module):
    """
    PEFT-compatible LeNALinear:
      - Keeps __init__(base_layer, module_key=None) (what PEFT expects)
      - Stores A/B as lora_A/lora_B so PEFT will mark them trainable
      - You can still refer to A/B if you like via aliases
      - Activation and gates are optional (can be Identity)
    """

    def __init__(self, base_layer: nn.Linear, module_key: Optional[str] = None):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"LeNALinear only supports nn.Linear, got {type(base_layer)}")

        self.base_layer = base_layer
        self.module_key = module_key or "<unknown>"

        # --- IMPORTANT: PEFT looks for these names for trainability ---
        self.lora_A = nn.ModuleDict()
        self.lora_B = nn.ModuleDict()

        # Optional: keep your old attribute names as aliases (same objects)
        self.A = self.lora_A
        self.B = self.lora_B

        # Activations / dropout / gate per adapter name
        self.act = nn.ModuleDict()
        self.drop = nn.ModuleDict()
        # Single selection gate g in [0,1]: interpolates z (linear/LoRA) <-> phi(z) (nonlinear),
        # via h = z + g*(phi(z) - z) = (1-g)*z + g*phi(z). Initialized closed => starts at LoRA.
        self.gate = nn.ModuleDict()
        self.steer = nn.ModuleDict()

        self.scaling: Dict[str, float] = {}
        self._active_adapter: Optional[str] = None

        self.norm_before_act = nn.ModuleDict()
        self.magnitude = nn.ParameterDict()
        # debug / logging
        self._forward_logged: Dict[str, bool] = {}
        self._dbg: Dict[str, Dict[str, bool]] = {}

        self.use_dora = False   # set per-adapter from cfg.lena_use_dora in add_adapter()

        w = self.base_layer.weight
        out_f, in_f = self.out_features, self.in_features

        if tuple(w.shape) == (out_f, in_f):
            self.fan_in_fan_out = True  # standard nn.Linear
        elif tuple(w.shape) == (in_f, out_f):
            self.fan_in_fan_out = False  # transposed storage
        else:
            # Fallback: default to False but warn loudly
            self.fan_in_fan_out = True

        self.init = False

    @property
    def in_features(self) -> int:
        return self.base_layer.in_features

    @property
    def out_features(self) -> int:
        return self.base_layer.out_features

    def set_active_adapter(self, name: Optional[str]):
        # PEFT will call this
        self._active_adapter = name

    def add_adapter(self, adapter_name: str, cfg: LeNAConfig):
        """
        PEFT calls this (usually adapter_name == "default").
        The key fix is: put A/B into lora_A/lora_B so PEFT will unfreeze them.
        """
        r = int(cfg.r)
        if r <= 0:
            raise ValueError("LeNAConfig.r must be > 0")

        # Create A/B
        A = nn.Linear(self.in_features, r, bias=False)
        B = nn.Linear(r, self.out_features, bias=False)

        # Original LoRA init style
        # nn.init.kaiming_uniform_(A.weight, a=5**0.5, nonlinearity='leaky_relu')
        nn.init.xavier_uniform_(A.weight)
        nn.init.zeros_(B.weight)

        # --- register under PEFT-recognized names ---
        self.lora_A[adapter_name] = A
        self.lora_B[adapter_name] = B

        # Make sure they are trainable (even if something else tries to freeze)
        for p in self.lora_A[adapter_name].parameters():
            p.requires_grad = True
        for p in self.lora_B[adapter_name].parameters():
            p.requires_grad = True

        # dropout
        lora_dropout = float(getattr(cfg, "lora_dropout", 0.0) or 0.0)
        self.drop[adapter_name] = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()

        # DoRA-style decomposition (LeNA-D). Default OFF so the nonlinearity's
        # contribution can be separated from DoRA in ablations.
        self.use_dora = bool(getattr(cfg, "lena_use_dora", False))

        # Normalize the code z before the activation (input conditioning for phi).
        self.use_norm_before_act = bool(getattr(cfg, "lena_norm_before_act", False))

        # scaling
        # This is for LoRA
        # self.scaling[adapter_name] = float(cfg.lora_alpha) / float(r)

        lena_nonlinear_scale = 1.0
        self.scaling[adapter_name] = float(cfg.lora_alpha) / float(r) * lena_nonlinear_scale

        # "token" = per-token LayerNorm (original); "shared" = one running scalar that
        # keeps relative token magnitudes. See SharedRMS for why the difference matters.
        norm_mode = str(getattr(cfg, "lena_norm_mode", "token")).lower()
        if norm_mode == "shared":
            self.norm_before_act[adapter_name] = SharedRMS()
        elif norm_mode == "token":
            self.norm_before_act[adapter_name] = nn.LayerNorm(r)
        else:
            raise ValueError(f"lena_norm_mode must be 'token' or 'shared', got {norm_mode!r}")

        # activation (can be identity). Built in RAW mode: the layer's selection gate
        # (below) does the linear<->nonlinear interpolation, so activations no longer
        # carry their own inconsistent identity-init / internal gating.
        act_kwargs = dict(cfg.lena_activation_kwargs or {})
        if str(cfg.lena_activation).lower() in ("fourier", "spline", "polynomial", "swish", "tanhres", "rankmix", "rankmix2", "bilinear", "afa", "aurora", "loran", "rankmixc", "aurorag"):
            act_kwargs["use_gate"] = "none"
        self.act[adapter_name] = make_lena_activation(
            kind=cfg.lena_activation,
            mode=cfg.lena_flex_mode,
            **act_kwargs,
        )
        if os.environ.get("LENA_STEER"):
            _p=int(os.environ.get("LENA_STEER_P", "4"))
            _initU=None
            if os.environ.get("LENA_STEER_SVD"):
                try:
                    _W=self.base_layer.weight.data.float()
                    _q=min(_p+4, min(_W.shape)-1)
                    _U,_S,_V=torch.svd_lowrank(_W, q=_q)
                    _initU=_U[:, :_p].contiguous()          # top-p left singular vectors
                except Exception as _e:
                    _initU=None
            self.steer[adapter_name] = SteerHead(
                self.out_features, r, p=_p,
                k=int(os.environ.get("LENA_STEER_K", "1")),
                init_U=_initU,
            )
            for p in self.steer[adapter_name].parameters():
                p.requires_grad = True

        # The Flex activations allocate their learnable parameters (spline knots_y,
        # Fourier coefficients, ...) lazily on the first forward -- which happens AFTER
        # the optimizer is constructed, so they never received a single update and the
        # "learned nonlinearity" was in fact a fixed one. Materialize them now, while
        # the module is still being built.
        #
        # 'global' -> (1,1,1) and 'dim' -> (1,1,C) are the shapes that do not depend on
        # the input's H/W, so a zero probe of width r produces exactly the right shape.
        flex_mode = str(cfg.lena_flex_mode).lower()
        if str(cfg.lena_activation).lower() != "identity":
            if flex_mode not in ("global", "dim"):
                raise ValueError(
                    f"lena_flex_mode={flex_mode!r} sizes the activation by sequence "
                    "length, so its parameters cannot be registered before the "
                    "optimizer is built; use 'global' or 'dim'."
                )
            with torch.no_grad():
                # same [..., H, W, C] layout _to_hwc() hands the activation at runtime
                probe = torch.zeros(1, 1, 1, r, dtype=A.weight.dtype, device=A.weight.device)
                self.act[adapter_name](probe)

        # weight_norm

        with torch.no_grad():
            # Compute effective LoRA weight matrix: B @ A
            lora_weight = B.weight @ A.weight  # [out_features, in_features]

            # Get base layer weight
            weight = dequantize_module_weight(self.base_layer)
            lora_weight = lora_weight.to(device=weight.device, dtype=weight.dtype)

            # Compute initial magnitude as the column-wise norm of (W + scaling * ΔW)
            weight_norm = self.get_weight_norm(weight, lora_weight, self.scaling[adapter_name])

            # Initialize magnitude parameter with proper shape [out_features]
            self.magnitude[adapter_name] = nn.Parameter(
                weight_norm.clone(),
                requires_grad=True
            )

        # ---- Selection gate ----
        # One gate g in [0,1] on the low-rank code z, at the configured placement
        # granularity (global/rank/token/voxel). Initialized near-CLOSED (g approx 0)
        # so LeNA starts exactly at LoRA and learns *where* to open nonlinearity.
        gate_type = str(getattr(cfg, "lena_gate_type", "none")).lower()
        gate_init = float(getattr(cfg, "lena_gate_init", -2.0))
        mode = str(getattr(cfg, "lena_gate_mode", "global")).lower()
        gate_strength = str(getattr(cfg, "gate_strength", "soft")).lower()
        init = 0.0 if gate_type == "rezero" else gate_init

        if gate_type != "none":
            self.gate[adapter_name] = Gate(
                gate_type=gate_type,  # type: ignore[arg-type]
                gate_mode=mode,
                init=init,
                dtype=A.weight.dtype,
                device=A.weight.device,
                gate_strength=gate_strength,
                code_dim=r,
            )
        else:
            self.gate[adapter_name] = nn.Identity()  # None gate == always-on nonlinearity (g=1)

        # debug flags
        self._dbg[adapter_name] = {
            "debug": bool(getattr(cfg, "lena_debug", False)),
            "verbose": bool(getattr(cfg, "lena_debug_verbose", False)),
            "forward": bool(getattr(cfg, "lena_debug_forward", False)),
            "forward_once": bool(getattr(cfg, "lena_debug_forward_once", True)),
            "check_nan": bool(getattr(cfg, "lena_debug_check_nan", False)),
        }
        self._forward_logged[adapter_name] = False

        # Every submodule above was constructed on the CPU. Upstream PEFT moves newly
        # injected LoRA modules onto the base layer's device; LeNA never did, so
        # loading a trained adapter onto a GPU model died on the first A(x) with
        # "found at least two devices, cuda:0 and cpu". Training happened to survive
        # because a later model.to(device) swept everything up, and the save/load test
        # ran entirely on CPU -- so this only ever showed up at evaluation time.
        base_w = getattr(self.base_layer, "weight", None)
        if base_w is not None:
            for mod in (self.lora_A[adapter_name], self.lora_B[adapter_name],
                        self.act[adapter_name], self.gate[adapter_name],
                        self.norm_before_act[adapter_name]):
                mod.to(device=base_w.device)
            self.magnitude[adapter_name].data = self.magnitude[adapter_name].data.to(base_w.device)

        if self._active_adapter is None:
            self._active_adapter = adapter_name

    def _pick_adapter(self, adapter_name: Optional[str]) -> Optional[str]:
        if adapter_name is not None:
            return adapter_name
        if self._active_adapter is not None:
            return self._active_adapter
        if len(self.lora_A) == 1:
            return next(iter(self.lora_A.keys()))
        return None

    def get_weight_norm(self, weight, lora_weight, scaling) -> torch.Tensor:
        # calculate L2 norm of weight matrix, column-wise
        # weight = transpose(weight, self.fan_in_fan_out)
        # lora_weight = transpose(lora_weight, self.fan_in_fan_out)

        if lora_weight.shape != weight.shape:
            lora_weight = transpose(lora_weight, True)

        # print("Wright=", lora_weight.shape, weight.shape)

        weight = weight + scaling * lora_weight
        weight_norm = torch.linalg.norm(weight, dim=1).to(weight.dtype)
        return weight_norm

    def forward(self, x: torch.Tensor, adapter_name: Optional[str] = None) -> torch.Tensor:
        y = self.base_layer(x)

        name = self._pick_adapter(adapter_name)
        if name is None or name not in self.lora_A:
            return y

        A = self.lora_A[name]
        B = self.lora_B[name]
        act = self.act[name]
        drop = self.drop[name]
        gate = self.gate[name]
        scale = self.scaling[name]

        # Low-rank code z = A x  (shape [..., r]).
        # The base layer may be fp16/bf16 while the adapter params are kept in fp32
        # (mixed-precision training), so run the whole adapter branch in A's dtype
        # and cast the delta back to y's dtype at the end.
        z = A(drop(x).to(A.weight.dtype))

        # Gated linear<->nonlinear interpolation in the code:
        #     h = z + g * (phi(z) - z) = (1-g) z + g phi(z)
        # g in [0,1] is the selection gate (None => always-on nonlinearity).
        if getattr(act, "kind", None) == "identity":
            h = z  # pure LoRA path (no nonlinearity requested)
        else:
            # Input conditioning for phi (does NOT touch the linear skip below).
            # Under fp16 autocast A's output comes back Half while the norm keeps fp32
            # params, and CUDA layer_norm does not promote for us -> run it in the norm's
            # own dtype and hand phi back a tensor shaped like z.
            if self.use_norm_before_act:
                norm = self.norm_before_act[name]
                nw = getattr(norm, "weight", None)
                zc = (norm(z.to(nw.dtype)).to(z.dtype) if nw is not None else norm(z))
            else:
                zc = z
            z_hwc, _, orig_ndim = _to_hwc(zc)
            # Custom ops (LayerNorm/spline) may upcast to float32 under autocast; keep the
            # whole interpolation in z's dtype so B(h) matches its weights (fp16/bf16).
            phi = _from_hwc(act(z_hwc), orig_ndim).to(z.dtype)
            phi = phi.clamp(-50.0, 50.0)  # numerical guard only (not a functional shaper)
            g = self._gate_value(gate, z)
            if g is not None:
                g = g.to(z.dtype)
            # skip uses raw z => gate closed recovers exact LoRA regardless of norm.
            h = phi if g is None else z + g * (phi - z)

        dz = B(h)
        if name in self.steer:
            st = self.steer[name]
            if next(st.parameters()).device != z.device:
                st.to(z.device)
            dz = dz + st(z).to(dz.dtype)

        if self.use_dora:
            # LeNA-D: DoRA-style magnitude/direction rescaling. NOTE: the direction norm
            # is computed from a *linearized* delta (B@A), an approximation when phi is
            # nonlinear. Kept as an optional variant, off by default.
            x_eye = torch.eye(A.weight.shape[1], device=A.weight.device, dtype=A.weight.dtype)
            lora_weight = B(A(x_eye))
            weight = dequantize_module_weight(self.base_layer).to(lora_weight.dtype)
            weight_norm = self.get_weight_norm(weight, lora_weight.detach(), scale).detach()
            mag_norm_scale = (self.magnitude[name] / weight_norm).view(1, -1).to(y.dtype)
            if self.base_layer.bias is not None:
                y = y - self.base_layer.bias
            return mag_norm_scale * y + mag_norm_scale * dz.to(y.dtype) * scale

        # Plain (non-DoRA) LeNA path: frozen output + scaled nonlinear low-rank delta.
        return y + (dz * scale).to(y.dtype)

    def _gate_value(self, gate: nn.Module, z: torch.Tensor) -> Optional[torch.Tensor]:
        """Return the selection gate g in [0,1] broadcastable to z, or None (always-on)."""
        if isinstance(gate, nn.Identity):
            return None
        return gate.value(z)


# ---------------------------------------------------------------------------
# Model-level utilities for the selection gate: L1 sparsity penalty (training)
# and a per-module openness report (the "where is nonlinearity used" analysis).
# ---------------------------------------------------------------------------
def lena_gate_l1(model: nn.Module, adapter_name: Optional[str] = None) -> Optional[torch.Tensor]:
    """Mean gate openness E[g] over all LeNALinear selection gates.

    Add `lambda * lena_gate_l1(model)` to the training loss to encourage most
    locations to stay linear (LoRA), so nonlinearity is spent only where it helps.
    Returns None if no active (materialized) gates exist yet.
    """
    total = None
    count = 0
    for m in model.modules():
        if not isinstance(m, LeNALinear):
            continue
        gates = m.gate.items() if adapter_name is None else [(adapter_name, m.gate.get(adapter_name))]
        for _, g in gates:
            if isinstance(g, Gate):
                o = g.openness()
                if o is not None:
                    s = o.sum()
                    total = s if total is None else total + s
                    count += o.numel()
    if total is None or count == 0:
        return None
    return total / count


def lena_gate_report(model: nn.Module) -> Dict[str, float]:
    """Map module_key -> mean gate openness in [0,1]. Feed to a heatmap to show
    which layers/positions opened the nonlinear path."""
    report: Dict[str, float] = {}
    for m in model.modules():
        if not isinstance(m, LeNALinear):
            continue
        for name, g in m.gate.items():
            if isinstance(g, Gate):
                o = g.openness()
                if o is not None:
                    report[f"{m.module_key}::{name}"] = float(o.mean().detach())
    return report
