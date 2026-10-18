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


class TwoPathTTT(nn.Module):
    """ViT^3-grounded two-pathway TTT adapter, made CAUSAL (fixes the future-leakage that broke the
    non-causal port on autoregressive generation). Global = causal SwiGLU inner-TTT: weights (w1,w2)
    fit on the CAUSAL prefix (k_1..k_i -> v) via a hand-derived one-step gradient (cumulative state,
    so position i only sees j<=i), applied to q -> (q w1*)*silu(q w2*). Local = causal depthwise
    conv1d (short-range). Codes concatenated and up-projected by Bt (init 0 => starts at base).
    Trained end-to-end with the task's normal loss (like ViT^3), differentiating through the
    closed-form inner step (no 2nd-order autograd)."""
    def __init__(self, d_in, d_out, r, lr=1.0, ksize=4):
        super().__init__()
        self.r = int(r); self.lr = float(lr); self.scale = float(r) ** -0.5; self.ksize = int(ksize)
        self.Wq = nn.Parameter(torch.randn(int(r), int(d_in)) * (float(d_in) ** -0.5))
        self.Wk = nn.Parameter(torch.randn(int(r), int(d_in)) * (float(d_in) ** -0.5))
        self.Wv = nn.Parameter(torch.randn(int(r), int(d_in)) * (float(d_in) ** -0.5))
        self.w1 = nn.Parameter(torch.empty(int(r), int(r))); nn.init.trunc_normal_(self.w1, std=0.02)
        self.w2 = nn.Parameter(torch.empty(int(r), int(r))); nn.init.trunc_normal_(self.w2, std=0.02)
        self.Wq2 = nn.Parameter(torch.randn(int(r), int(d_in)) * (float(d_in) ** -0.5))
        self.conv = nn.Parameter(torch.zeros(int(r), 1, int(ksize)))   # depthwise causal kernel, init 0
        self.Bt = nn.Parameter(torch.zeros(int(d_out), 2 * int(r)))    # up-proj, init 0 => safe start

    def _causal_swiglu(self, q, k, v):
        z1 = torch.matmul(k, self.w1); z2 = torch.matmul(k, self.w2)
        sig = torch.sigmoid(z2); a = z2 * sig
        e = -v * self.scale
        ea = e * a
        eb = e * z1 * (sig * (1.0 + z2 * (1.0 - sig)))
        # causal cumulative inner-gradient states (position i sees only j<=i):
        Nn = k.shape[-2]
        cnt = torch.arange(1, Nn + 1, device=k.device, dtype=k.dtype).view(1, Nn, 1, 1)
        # ViT3 fix 1: prefix-MEAN gradient (ViT3 divides by N; causal => divide by i) -- keeps the
        # inner update from exploding as the prefix grows.
        S1 = torch.cumsum(torch.einsum('bnr,bns->bnrs', k, ea), dim=1) / cnt
        S2 = torch.cumsum(torch.einsum('bnr,bns->bnrs', k, eb), dim=1) / cnt
        # ViT3 fix 2: gradient CLIPPING g/(||g||+1) 'for stability' (per position, over dim -2).
        S1 = S1 / (S1.norm(dim=-2, keepdim=True) + 1.0)
        S2 = S2 / (S2.norm(dim=-2, keepdim=True) + 1.0)
        qw1 = torch.matmul(q, self.w1) - self.lr * torch.einsum('bnr,bnrs->bns', q, S1)
        qw2 = torch.matmul(q, self.w2) - self.lr * torch.einsum('bnr,bnrs->bns', q, S2)
        return qw1 * torch.nn.functional.silu(qw2)

    def _causal_conv(self, q2):
        xc = q2.transpose(1, 2)                                   # [B,r,N]
        xc = torch.nn.functional.pad(xc, (self.ksize - 1, 0))    # left pad => causal
        o = torch.nn.functional.conv1d(xc, self.conv.to(xc.dtype), groups=self.r)
        return o.transpose(1, 2)                                  # [B,N,r]

    def forward(self, x, z=None):
        sq = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            if z is not None:
                z = z.unsqueeze(0)
            sq = True
        if z is not None:
            # SHARED-CODE mode: reuse LoRA's own r-dim code z as q=k=v (no Wq/Wk/Wv) -> TTT refines
            # LoRA's learned code. Param-efficient + semantically grounded.
            zc = z.to(self.w1.dtype)
            q = k = v = zc
            o1 = self._causal_swiglu(q, k, v)
            o2 = self._causal_conv(zc)
        else:
            xf = x.to(self.Wq.dtype)
            q = torch.matmul(xf, self.Wq.t()); k = torch.matmul(xf, self.Wk.t()); v = torch.matmul(xf, self.Wv.t())
            o1 = self._causal_swiglu(q, k, v)
            o2 = self._causal_conv(torch.matmul(xf, self.Wq2.t()))
        o = torch.cat([o1, o2], dim=-1)                           # [B,N,2r]
        delta = torch.matmul(o, self.Bt.t())
        if sq:
            delta = delta.squeeze(0)
        return delta.to(x.dtype)


class TTTHead(nn.Module):
    """Simplified TTT: cross-token similarity mixing of the low-rank code (one gram matrix).
    k = A x (d->r);  c_i = (1/N) sum_j <k_j,k_i> k_j = (1/N) k (K^T K)_i ;  delta = gamma * B_t c.
    This is the closed form of ONE test-time step of a linear inner model, stripped of all
    machinery (no SwiGLU/hand-grad/Q,K,V) -- token i's code is a similarity-weighted average of
    the whole sequence's codes. Non-causal (full-sequence gram): judge by LIKELIHOOD eval, where
    it is fully active (no autoregressive-decode mismatch). gamma init 0 => exact base/LoRA start.
    Cheap: gram is r x r. Output in col(B_t)."""
    def __init__(self, d_in, d_out, r, lr=1.0):
        super().__init__()
        self.A = nn.Parameter(torch.randn(int(r), int(d_in)) * (float(d_in) ** -0.5))
        self.Bt = nn.Parameter(torch.randn(int(d_out), int(r)) * (float(r) ** -0.5))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        sq = False
        if x.dim() == 2:
            x = x.unsqueeze(0); sq = True
        xf = x.to(self.A.dtype)
        k = torch.matmul(xf, self.A.t())                 # [B,N,r]
        N = k.shape[-2]
        gram = torch.matmul(k.transpose(-2, -1), k) / float(N)   # [B,r,r]
        c = torch.matmul(k, gram)                        # [B,N,r]  cross-token mixed code
        delta = torch.matmul(c, self.Bt.t())             # [B,N,d_out]
        out = self.gamma.to(delta.dtype) * delta
        if sq:
            out = out.squeeze(0)
        return out.to(x.dtype)


class OutGate(nn.Module):
    """Input-dependent soft Householder reflection of the LoRA write.
    delta = dz - 2*alpha*v_hat*(v_hat . dz),  v = Wv z,  v_hat = v/||v||.
    alpha (scalar, init 0) => exact LoRA at start AND non-zero gradient (v_hat defined since Wv is
    random-init) -> avoids the DEAD SADDLE of zero-init Wv (reflection is quadratic in v near 0).
    alpha=1 => full norm-preserving reflection; alpha in (0,1) => partial suppression of the
    input-selected write direction v_hat."""
    def __init__(self, d_out, r):
        super().__init__()
        self.Wv = nn.Parameter(torch.randn(int(d_out), int(r)) * (float(r) ** -0.5))
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, z, dz):
        v = torch.matmul(z.to(self.Wv.dtype), self.Wv.transpose(0, 1)).to(dz.dtype)
        vhat = v / (v.norm(dim=-1, keepdim=True) + 1e-6)
        proj = (vhat * dz).sum(dim=-1, keepdim=True)
        return dz - 2.0 * self.alpha.to(dz.dtype) * vhat * proj


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


def _causal_mean(x):
    """Causal running mean over the sequence dim (-2): xbar_i = mean(x_1..x_i). Leak-free
    'global context so far'. Decode note: with KV-cache a single-token forward sees N=1 so
    xbar=x (global pathway sees the token, local sees ~0); judge two-pathway by LIKELIHOOD eval
    (full sequence) where it is fully active."""
    if x.dim() < 2:
        return x
    cs = x.cumsum(dim=-2)
    n = x.shape[-2]
    cnt = torch.arange(1, n + 1, device=x.device, dtype=cs.dtype).view(-1, 1)
    return cs / cnt


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
        self.outgate = nn.ModuleDict()
        self.ttt = nn.ModuleDict()
        self.twopttt = nn.ModuleDict()
        self.lora_Ag = nn.ModuleDict()
        self.lora_Bg = nn.ModuleDict()
        self.lora_A2 = nn.ModuleDict()
        self.iq_lam = nn.ParameterDict()
        self.pclora_lam = nn.ParameterDict()

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
        if os.environ.get("LENA_PCLORA"):
            self.pclora_lam[adapter_name] = nn.Parameter(torch.zeros(1))  # lam init 0 => exact LoRA start
        if os.environ.get("LENA_IQ"):
            A2 = nn.Linear(self.in_features, r, bias=False)
            nn.init.xavier_uniform_(A2.weight)
            self.lora_A2[adapter_name] = A2
            self.iq_lam[adapter_name] = nn.Parameter(torch.zeros(1) + float(os.environ.get("LENA_IQ_LAM","0.0")))  # lambda init (env; 0 => exact LoRA start, >0 => warm quadratic)
            for p in A2.parameters(): p.requires_grad = True
        if os.environ.get("LENA_TWOPATH"):
            Ag = nn.Linear(self.in_features, r, bias=False)
            Bg = nn.Linear(r, self.out_features, bias=False)
            nn.init.xavier_uniform_(Ag.weight); nn.init.zeros_(Bg.weight)
            self.lora_Ag[adapter_name] = Ag
            self.lora_Bg[adapter_name] = Bg
            for p in Ag.parameters(): p.requires_grad = True
            for p in Bg.parameters(): p.requires_grad = True

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
        if os.environ.get("LENA_OUTGATE"):
            self.outgate[adapter_name] = OutGate(self.out_features, r)
            for p in self.outgate[adapter_name].parameters():
                p.requires_grad = True
        if os.environ.get("LENA_TTT"):
            self.ttt[adapter_name] = TTTHead(self.in_features, self.out_features, r,
                                             lr=float(os.environ.get("LENA_TTT_LR", "1.0")))
            for p in self.ttt[adapter_name].parameters():
                p.requires_grad = True
        if os.environ.get("LENA_2PTTT"):
            self.twopttt[adapter_name] = TwoPathTTT(self.in_features, self.out_features, r,
                                                    lr=float(os.environ.get("LENA_2PTTT_LR", "1.0")))
            for p in self.twopttt[adapter_name].parameters():
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

    def _phi(self, name, z, act, gate):
        """z -> h : the AuroRA-style nonlinear-code step (norm -> act -> gated linear/nonlinear
        interpolation), shared by both pathways so each is 'like AuroRA'."""
        if getattr(act, "kind", None) == "identity":
            return z
        if self.use_norm_before_act:
            norm = self.norm_before_act[name]
            nw = getattr(norm, "weight", None)
            zc = (norm(z.to(nw.dtype)).to(z.dtype) if nw is not None else norm(z))
        else:
            zc = z
        z_hwc, _, orig_ndim = _to_hwc(zc)
        phi = _from_hwc(act(z_hwc), orig_ndim).to(z.dtype)
        phi = phi.clamp(-50.0, 50.0)
        g = self._gate_value(gate, z)
        if g is not None:
            g = g.to(z.dtype)
        return phi if g is None else z + g * (phi - z)

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
        # Two-pathway (LENA_TWOPATH): global = phi on the causal-mean context (captures the
        # shared/task-level structure), local = phi on the per-token deviation from that mean
        # (captures the individual token). Both AuroRA-style (shared _phi). Bg init 0 => two-path
        # starts at exact base.
        if name in self.lora_Ag:
            xbar = _causal_mean(x)
            z = A(drop(x - xbar).to(A.weight.dtype))
            dz = B(self._phi(name, z, act, gate))
            Ag = self.lora_Ag[name]; Bg = self.lora_Bg[name]
            if Ag.weight.device != x.device:
                Ag.to(x.device); Bg.to(x.device)
            zg = Ag(drop(xbar).to(Ag.weight.dtype))
            dz = dz + Bg(self._phi(name, zg, act, gate))
        else:
            z = A(drop(x).to(A.weight.dtype))
            if name in self.lora_A2:
                A2 = self.lora_A2[name]
                if A2.weight.device != x.device:
                    A2.to(x.device); self.iq_lam[name].data = self.iq_lam[name].data.to(x.device)
                z2 = A2(drop(x).to(A2.weight.dtype))
                z = z + self.iq_lam[name].to(z.dtype) * (z * z2)   # input-dependent quadratic code
            if name in self.pclora_lam:
                # Prompt-conditioned (TTT idea, decode-safe): causal-prefix associative-memory
                # fit MODULATING the code. corr_i = (1/i) sum_{j<=i} <z_j,z_i> z_j =
                # (1/i)(cumsum_j z_j z_j^T) z_i. Causal => decode-safe. lam init 0 => exact LoRA.
                if z.dim() >= 3:
                    zc = z.to(torch.float32)
                    S = torch.cumsum(torch.einsum('bnr,bns->bnrs', zc, zc), dim=1)
                    n = zc.shape[-2]
                    cnt = torch.arange(1, n + 1, device=zc.device, dtype=torch.float32).view(-1, 1)
                    corr = torch.einsum('bnr,bnrs->bns', zc, S) / cnt
                    z = z + self.pclora_lam[name].to(z.dtype) * corr.to(z.dtype)
            dz = B(self._phi(name, z, act, gate))
        if name in self.steer:
            st = self.steer[name]
            if next(st.parameters()).device != z.device:
                st.to(z.device)
            dz = dz + st(z).to(dz.dtype)
        if name in self.outgate:
            og = self.outgate[name]
            if og.Wv.device != z.device:
                og.to(z.device)
            dz = og(z, dz)
        if name in self.ttt:
            tt = self.ttt[name]
            if tt.A.device != x.device:
                tt.to(x.device)
            dz = dz + tt(x).to(dz.dtype)
        if name in self.twopttt:
            tp = self.twopttt[name]
            if tp.Wq.device != x.device:
                tp.to(x.device)
            if os.environ.get("LENA_2PTTT_SHAREZ"):
                dz = dz + tp(x, z=z).to(dz.dtype)   # reuse LoRA code z as q=k=v
            else:
                dz = dz + tp(x).to(dz.dtype)

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
