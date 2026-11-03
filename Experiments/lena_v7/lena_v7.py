"""LeNA v7: a low-rank adapter whose per-channel curve is fitted from the current input and shrunk toward a learned
prior (paper Sec. 3, Algorithm 1). Pure PyTorch; no peft dependency.

    u = A1 x, v = A2 x                       two reads, (..., N, r)
    n, m, e  = spline statistics of (u, v)   tridiagonal Gram, K knots
    J        = retained knots (LDL^T pivots > tau, restart after a removed knot)
    Delta    = argmin LS correction on J toward rho
    s        = (1 - gamma / F)_+,  gamma = exp(omega) * (p-2)/p * nu/(nu+2)
    c        = phi_{rho + s Delta}(u),   h = W x + (alpha / r) B c

Modes: "lena" (James-Stein step), "fixed" (s = 0, shared curve), "plain" (s = 1, plain retained-knot fit).
An affine per-input fit is mode "plain" with K = 2. Statistics, retention, solves and the step run in fp32.
"""
import math
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

FP32_TINY = torch.finfo(torch.float32).tiny


# ---------------------------------------------------------------------------------------------------------------
# Spline basis and statistics
# ---------------------------------------------------------------------------------------------------------------
def hat_index(u, lo, hi, K):
    """Left knot index j in [0, K-2] and weight lam of knot j+1; lam is unclamped outside the grid (linear
    extrapolation of the first/last segment)."""
    h = (hi - lo) / (K - 1)
    z = (u - lo) / h
    j = z.floor().clamp(0, K - 2)
    lam = z - j
    return j.long(), lam


def spline_eval(theta, j, lam):
    """theta (..., K) broadcast against j, lam (...). Returns theta_j (1-lam) + theta_{j+1} lam."""
    theta = theta.expand(*j.shape, theta.shape[-1])
    tj = torch.gather(theta, -1, j.unsqueeze(-1)).squeeze(-1)
    tj1 = torch.gather(theta, -1, (j + 1).unsqueeze(-1)).squeeze(-1)
    return tj * (1 - lam) + tj1 * lam


def stats_dense(j, lam, v, w, K):
    """Per-token contributions as dense (..., K) tensors. j, lam, v, w: (..., ) with w the token weight (0 = pad).
    Returns diag (..., K), off (..., K-1), m (..., K), e (...)."""
    a, b = (1 - lam) * w, lam * w                                    # weighted basis values at j and j+1
    oh = F.one_hot(j, K).to(lam.dtype)                               # (..., K)
    oh1 = F.one_hot(j + 1, K).to(lam.dtype)
    diag = oh * (a * (1 - lam)).unsqueeze(-1) + oh1 * (b * lam).unsqueeze(-1)
    off = (oh[..., :-1] * (a * lam).unsqueeze(-1))                   # n_{j,j+1}
    m = oh * (a * v).unsqueeze(-1) + oh1 * (b * v).unsqueeze(-1)
    e = w * v * v
    return diag, off, m, e


# ---------------------------------------------------------------------------------------------------------------
# Retained-knot LDL^T solve (vectorised over leading dims, sequential over K)
# ---------------------------------------------------------------------------------------------------------------
def retained_solve(diag, off, rhs, tau):
    """Pivot-retention rule of App. A.1 and the solve n_JJ Delta_J = rhs_J (Delta = 0 off J).
    diag, rhs (..., K); off (..., K-1). Returns Delta (..., K), retained mask (..., K) bool."""
    K = diag.shape[-1]
    piv, z, ret, L = [], [], [], [None]
    prev_d = prev_z = prev_r = None
    for k in range(K):
        if k == 0:
            d = diag[..., 0]
            zk = rhs[..., 0]
            lk = None
        else:
            c = torch.where(prev_r, off[..., k - 1], torch.zeros_like(off[..., k - 1]))   # coupling only to retained
            lk = c / prev_d
            d = diag[..., k] - lk * c
            zk = rhs[..., k] - lk * prev_z
        r = d > tau
        d = torch.where(r, d, torch.ones_like(d))                    # removed knot: unit pivot, zero rhs, no coupling
        zk = torch.where(r, zk, torch.zeros_like(zk))
        if k > 0:
            L.append(lk)
        piv.append(d); z.append(zk); ret.append(r)
        prev_d, prev_z, prev_r = d, zk, r
    # back substitution: Delta_k = z_k / d_k - L_{k+1} Delta_{k+1}, with L_{k+1} zero unless k and k+1 both retained
    delta = [None] * K
    delta[K - 1] = torch.where(ret[K - 1], z[K - 1] / piv[K - 1], torch.zeros_like(z[K - 1]))
    for k in range(K - 2, -1, -1):
        lnext = torch.where(ret[k + 1], L[k + 1], torch.zeros_like(L[k + 1]))
        dk = z[k] / piv[k] - lnext * delta[k + 1]
        delta[k] = torch.where(ret[k], dk, torch.zeros_like(dk))
    return torch.stack(delta, -1), torch.stack(ret, -1)


def tri_mv(diag, off, x):
    """Tridiagonal (symmetric) matrix-vector product."""
    y = diag * x
    y = y + F.pad(off * x[..., 1:], (0, 1)) + F.pad(off * x[..., :-1], (1, 0))
    return y


def prefix_stats(u, v, w, lo, hi, K):
    """Causal statistics: for every position, the sums over the prefix. Returns j, lam, diag, off, m, e, ntok."""
    j, lam = hat_index(u, lo, hi, K)
    diag, off, m, e = stats_dense(j, lam, v, w, K)
    return j, lam, diag.cumsum(1), off.cumsum(1), m.cumsum(1), e.cumsum(1), w.cumsum(1)


def pooled_stats(u, v, w, lo, hi, K):
    """Bidirectional statistics: sums over all positions of each input."""
    j, lam = hat_index(u, lo, hi, K)
    diag, off, m, e = stats_dense(j, lam, v, w, K)
    return j, lam, diag.sum(1), off.sum(1), m.sum(1), e.sum(1), w.sum(1)


def fit_and_step(diag, off, m, e, ntok, rho, omega, tau, mode):
    """Channel step of Algorithm 1. All inputs fp32; leading dims arbitrary, rho broadcastable to (..., K).
    Returns theta (..., K) and diagnostics (s, p, F)."""
    if mode == "fixed":
        theta = rho.expand(m.shape)
        z = torch.zeros(m.shape[:-1], device=m.device)
        return theta, dict(s=z, p=z, F=z)
    rhs = m - tri_mv(diag, off, rho)
    delta, ret = retained_solve(diag, off, rhs, tau)
    p = ret.sum(-1).to(m.dtype)
    nu = ntok - p
    if mode == "plain":
        return rho + delta, dict(s=torch.ones_like(p), p=p, F=torch.zeros_like(p))
    S = (delta * tri_mv(diag, off, delta)).sum(-1)
    V = (e - 2 * (rho * m).sum(-1) + (rho * tri_mv(diag, off, rho)).sum(-1) - S).clamp_min(0)
    ok = (p >= 3) & (nu >= 4) & (S > 0)
    nu_s = torch.where(ok, nu, torch.full_like(nu, 4.0))
    p_s = torch.where(ok, p, torch.full_like(p, 3.0))
    S_s = torch.where(ok, S, torch.ones_like(S))
    sig2 = torch.maximum(torch.maximum(V / nu_s, 1e-9 * e / ntok.clamp_min(1)), torch.full_like(V, FP32_TINY))
    gamma = torch.exp(omega) * (p_s - 2) * nu_s / (p_s * (nu_s + 2))
    s = (1 - gamma * p_s * sig2 / S_s).clamp_min(0)
    s = torch.where(ok, s, torch.zeros_like(s))
    Fstat = torch.where(ok, S_s / (p_s * sig2), torch.zeros_like(S))
    return rho + s.unsqueeze(-1) * delta, dict(s=s, p=p, F=Fstat)


_COMPILE = {"on": False, "fns": {}, "dynamic": True}


def set_compile(on=True, dynamic=True):
    """torch.compile the pure tensor functions of the layer (statistics, fit and step, curve evaluation). The layer
    otherwise launches a few hundred small kernels per call, which dominates decoding."""
    _COMPILE["on"] = bool(on)
    if dynamic != _COMPILE["dynamic"]:
        _COMPILE["dynamic"], _COMPILE["fns"] = dynamic, {}


def _fn(name):
    f = globals()[name]
    if not _COMPILE["on"]:
        return f
    if name not in _COMPILE["fns"]:
        _COMPILE["fns"][name] = torch.compile(f, dynamic=_COMPILE["dynamic"])
    return _COMPILE["fns"][name]


_TRITON = {"on": False}


def set_triton(on=True):
    """Fused Triton forward/backward (lena_triton.py) for mode "lena" with the learned or identity prior, causal or
    pooled, CUDA fp32. Other modes, u_pred / predictor, decoding state, stats record/inject and log_stats keep the
    PyTorch path."""
    _TRITON["on"] = bool(on)


# ---------------------------------------------------------------------------------------------------------------
# Context: padding mask, causality and decoding state, shared by every LeNA layer of a model
# ---------------------------------------------------------------------------------------------------------------
class _Ctx:
    mask = None        # (B, N) 1 for real tokens, 0 for padding; None = all real
    causal = None      # None -> use the layer default
    stream = False     # causal decoding with a KV cache: layers carry prefix statistics across calls
    stream_layers = []
    reordering = False
    # diagnostics (evaluation only)
    force_mode = None  # "fixed" (s = 0) | "plain" (s = 1) | "lena": overrides every layer's mode
    force_prior = None  # "identity": evaluate with the identity ordinates instead of the learned prior
    stats = None       # "record": keep each layer's statistics at the last position in L._rec
                       # "inject": use L._inj (B or 1, r, .) at every position instead of the input's own


_CTX = _Ctx()


@contextmanager
def lena_override(mode=None, prior=None, stats=None):
    """Evaluation-time interventions for the element checks (no effect on training)."""
    old = (_CTX.force_mode, _CTX.force_prior, _CTX.stats)
    _CTX.force_mode, _CTX.force_prior, _CTX.stats = mode, prior, stats
    try:
        yield
    finally:
        _CTX.force_mode, _CTX.force_prior, _CTX.stats = old


@contextmanager
def lena_context(attention_mask=None, causal=None):
    old = (_CTX.mask, _CTX.causal)
    _CTX.mask, _CTX.causal = attention_mask, causal
    try:
        yield
    finally:
        _CTX.mask, _CTX.causal = old


def lena_layers(model):
    return [m for m in model.modules() if isinstance(m, LeNALinear)]


@contextmanager
def lena_stream(model):
    """Wrap one model.generate(...) call that uses a KV cache. Resets the prefix statistics before and after."""
    layers = lena_layers(model)
    for L in layers:
        L._state = None
    _CTX.stream, _CTX.stream_layers = True, layers
    try:
        yield
    finally:
        _CTX.stream, _CTX.stream_layers = False, []
        for L in layers:
            L._state = None


def _reorder_lena(beam_idx):
    for L in getattr(_CTX, "stream_layers", []):
        L.reorder_state(beam_idx)


_PATCHED = False


def _patch_cache_reorder():
    """Beam search reorders the KV cache through Cache.reorder_cache (transformers 4.x and 5.x); reorder the LeNA
    prefix statistics in the same call so that the two never diverge."""
    global _PATCHED
    if _PATCHED:
        return
    try:
        import transformers.cache_utils as cu
    except ImportError:
        return
    for obj in vars(cu).values():
        if isinstance(obj, type) and "reorder_cache" in obj.__dict__:
            orig = obj.__dict__["reorder_cache"]

            def wrapped(self, beam_idx, _orig=orig):
                outer = not _CTX.reordering                       # Cache -> per-layer calls: reorder LeNA once
                if outer:
                    _reorder_lena(beam_idx)
                    _CTX.reordering = True
                try:
                    return _orig(self, beam_idx)
                finally:
                    if outer:
                        _CTX.reordering = False
            obj.reorder_cache = wrapped
    _PATCHED = True


def install_lena_hooks(model):
    """Read attention_mask from every top-level forward (training and generation) into the LeNA context, and
    reorder the decoding statistics together with the KV cache during beam search."""
    def pre_hook(mod, args, kwargs):
        mask = kwargs.get("attention_mask")
        ids = kwargs.get("input_ids", args[0] if args else None)
        if ids is None:
            ids = kwargs.get("inputs_embeds")
        if mask is not None and ids is not None:
            mask = mask[:, -ids.shape[1]:]                           # new tokens only (decoding passes full mask)
        _CTX.mask = mask
        return args, kwargs

    handle = model.register_forward_pre_hook(pre_hook, with_kwargs=True)
    _patch_cache_reorder()

    def _reorder_cache(past_key_values, beam_idx):
        if hasattr(past_key_values, "reorder_cache"):              # patched: reorders LeNA as well
            past_key_values.reorder_cache(beam_idx)
            return past_key_values
        _reorder_lena(beam_idx)                                      # legacy tuple caches
        return tuple(tuple(t.index_select(0, beam_idx.to(t.device)) for t in layer) for layer in past_key_values)

    model._reorder_cache = _reorder_cache
    return handle


# ---------------------------------------------------------------------------------------------------------------
# Layer
# ---------------------------------------------------------------------------------------------------------------
def _mlp(d_in, d_out, hidden=64):
    net = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, d_out))
    nn.init.zeros_(net[-1].weight)
    nn.init.zeros_(net[-1].bias)                                     # starts at the prior
    return net


class LeNALinear(nn.Module):
    """Wraps a frozen nn.Linear; input (B, N, d_in) or (N, d_in).

    mode   lena | fixed (s=0) | plain (s=1) | predictor (MLP from (n, m, e) to the curve; component control)
    prior  learned | identity (frozen identity ordinates) | u_pred (learned prior plus an MLP of first-read
           moments; independent of the response, as the theorem requires)
    learn_omega   False keeps omega = 0, i.e. the James-Stein scale
    freeze_A2     keeps the second read at its random initialization
    dora          LeNA-D: DoRA magnitude/direction around the code map (direction norm uses B A1, detached)
    """

    def __init__(self, base: nn.Linear, r=8, alpha=16, K=16, lo=-3.0, hi=3.0, tau=1.0, mode="lena",
                 prior="learned", learn_omega=True, freeze_A2=False, dora=False, causal=False, dropout=0.0,
                 use_checkpoint=True, log_stats=False, base_nl=None):
        super().__init__()
        assert mode in ("lena", "fixed", "plain", "predictor") and prior in ("learned", "identity", "u_pred")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        d_in, d_out = base.in_features, base.out_features
        self.r, self.K, self.lo, self.hi, self.tau = r, K, lo, hi, tau
        self.scaling = alpha / r
        self.mode, self.prior, self.causal = mode, prior, causal
        self.use_checkpoint, self.log_stats = use_checkpoint, log_stats
        dev = base.weight.device
        self.A1 = nn.Parameter(torch.empty(r, d_in, device=dev))
        self.A2 = nn.Parameter(torch.empty(r, d_in, device=dev), requires_grad=not freeze_A2)
        self.B = nn.Parameter(torch.zeros(d_out, r, device=dev))
        nn.init.xavier_uniform_(self.A1)
        nn.init.xavier_uniform_(self.A2)
        knots = torch.linspace(lo, hi, K, device=dev)
        self.rho = nn.Parameter(knots.repeat(r, 1), requires_grad=prior != "identity")   # identity init
        self.omega = nn.Parameter(torch.zeros((), device=dev), requires_grad=learn_omega)
        self.anl = None
        if base_nl == "aurora":                      # LeNA on AuroRA: learned shared non-linearity, per-input residual fit
            self.anl = ANL(r, r).to(dev)
            self.rho = nn.Parameter(torch.zeros(r, K, device=dev), requires_grad=False)
        self.prior_net = _mlp(7, K).to(dev) if prior == "u_pred" else None
        self.stat_net = _mlp(3 * K, K).to(dev) if mode == "predictor" else None
        if dora:
            self.magnitude = nn.Parameter(base.weight.detach().float().norm(dim=1).to(dev))
        else:
            self.magnitude = None
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.last = None
        self._state = None
        self.stat_drop = 0.0   # training only: each sample keeps a random fraction in [1 - stat_drop, 1] of its tokens

    def extra_repr(self):
        return (f"r={self.r}, K={self.K}, grid=[{self.lo},{self.hi}], tau={self.tau}, mode={self.mode}, "
                f"prior={self.prior}, causal={self.causal}, dora={self.magnitude is not None}")

    # -- decoding state ------------------------------------------------------------------------------------------
    def reorder_state(self, idx):
        if self._state is not None:
            self._state = tuple(t.index_select(0, idx.to(t.device)) for t in self._state)

    # -- curve per input / prefix --------------------------------------------------------------------------------
    @staticmethod
    def _u_moments(u, w):
        """Seven weighted first-read moments per token, (..., 7) before pooling (same as the controlled check)."""
        return torch.stack([u, u * u, u ** 3, u.abs(), u.clamp_min(0), (u > 1).to(u.dtype), u * u], -1) * \
            w.unsqueeze(-1)

    def _pool_u(self, mom, ntok):
        mean = mom / ntok.clamp_min(1).unsqueeze(-1)
        std = (mean[..., 6] - mean[..., 0] ** 2).clamp_min(0).sqrt()
        return torch.cat([mean[..., :6], std.unsqueeze(-1)], -1)

    def _codes(self, u, v, w):
        """u, v, w: (B, N, r) (w = token weight). Returns codes (B, N, r) and diagnostics."""
        K = self.K
        rho, omega = self.rho.to(u.dtype), self.omega.to(u.dtype)
        if self.training and self.stat_drop > 0:       # thin the statistics so the step meets weak evidence too
            keep = 1 - self.stat_drop * torch.rand(w.shape[0], 1, 1, device=w.device, dtype=w.dtype)
            w = w * (torch.rand(w.shape[0], w.shape[1], 1, device=w.device, dtype=w.dtype) < keep).to(w.dtype)
        if _CTX.force_prior == "identity":
            rho = torch.linspace(self.lo, self.hi, K, device=u.device, dtype=u.dtype).expand_as(rho)
        mode = _CTX.force_mode or self.mode
        causal = self.causal if _CTX.causal is None else _CTX.causal
        if (_TRITON["on"] and mode == "lena" and self.prior_net is None and u.is_cuda and u.dtype == torch.float32
                and _CTX.stats is None and not self.log_stats and not (causal and _CTX.stream and not self.training)
                and not (self.training and getattr(self, "gate_drop", 0.0) > 0)):   # gate_drop lives in the PyTorch path
            # patched layers (vision/lena_mri_ext.py) keep the PyTorch path, except LeNA-S (n_eff only), which the
            # fused kernel implements itself (design effect deff passed in, see lena_triton.py NEFF)
            pt = _TRITON.get("patched")
            if not pt:
                from lena_triton import fused_codes
                return fused_codes(u, v, w, rho, omega, self.lo, self.hi, self.tau, causal)
            if _TRITON.get("neff_ok") is not None and _TRITON["neff_ok"]():
                from lena_triton import fused_codes
                return _TRITON["neff_codes"](self, fused_codes, u, v, w, rho, omega, causal)
        mom = self._u_moments(u, w) if self.prior_net is not None else None
        if causal:
            j, lam, *acc = _fn("prefix_stats")(u, v, w, self.lo, self.hi, K)
            if mom is not None:
                acc.append(mom.cumsum(1))
            if _CTX.stream and not self.training:
                if self._state is not None:
                    acc = [a + s.unsqueeze(1) for a, s in zip(acc, self._state)]
                self._state = tuple(a[:, -1].detach() for a in acc)
            if _CTX.stats == "record":
                self._rec = tuple(a[:, -1].detach() for a in acc)
            elif _CTX.stats == "inject":
                acc = [s.to(a.dtype).unsqueeze(1).expand_as(a) for a, s in zip(acc, self._inj)]
            diag, off, m, e, ntok = acc[:5]
            mom = acc[5] if mom is not None else None
        else:
            j, lam, diag, off, m, e, ntok = _fn("pooled_stats")(u, v, w, self.lo, self.hi, K)      # (B, r, ·)
            if _CTX.stats == "record":                     # per-input sums, to be pooled across inputs outside
                self._rec = tuple(a.detach() for a in (diag, off, m, e, ntok))
            elif _CTX.stats == "inject":                   # statistics pooled over other (unlabeled) inputs
                diag, off, m, e, ntok = [s_.to(a.dtype).expand_as(a) for s_, a in zip(self._inj, (diag, off, m, e, ntok))]
            mom = mom.sum(1) if mom is not None else None
        if mom is not None:
            rho = rho + self.prior_net(self._pool_u(mom, ntok).to(self.prior_net[0].weight.dtype)).to(u.dtype)
        if mode == "predictor":
            n1 = ntok.clamp_min(1).unsqueeze(-1)
            feat = torch.cat([diag / n1, F.pad(off, (0, 1)) / n1, m / n1], -1)
            theta = rho + self.stat_net(feat.to(self.stat_net[0].weight.dtype)).to(u.dtype)
            z = torch.zeros_like(ntok)
            d = dict(s=z, p=z, F=z)
        else:
            theta, d = _fn("fit_and_step")(diag, off, m, e, ntok, rho, omega, self.tau, mode)
            if self.training and getattr(self, "gate_drop", 0.0) > 0:   # train the shared-curve path too (shift gate)
                keep = (torch.rand(theta.shape[0], *([1] * (theta.dim() - 1)), device=theta.device) >= self.gate_drop).to(theta.dtype)
                theta = rho + keep * (theta - rho)
        if self.log_stats:
            d = dict(d, theta=theta.detach(), u=u.detach(), v=v.detach())
        if not causal:
            theta = theta.unsqueeze(1)
        return _fn("spline_eval")(theta, j, lam), d

    def forward(self, x):
        squeeze = x.dim() == 2
        if squeeze:
            x = x.unsqueeze(0)
        xd = self.drop(x)
        sd = torch.promote_types(xd.dtype, torch.float32)           # fp32 statistics (fp64 kept for tests)
        u = F.linear(xd, self.A1.to(xd.dtype)).to(sd)
        v = F.linear(xd, self.A2.to(xd.dtype)).to(sd)
        Bsz, N, r = u.shape
        mask = _CTX.mask
        if mask is not None and tuple(mask.shape[:2]) == (Bsz, N):
            w = mask.to(sd).unsqueeze(-1).expand(Bsz, N, r)
        else:
            w = torch.ones_like(u)
        g = self.anl(u) if self.anl is not None else None
        vt = v - g if g is not None else v
        if self.use_checkpoint and self.training and torch.is_grad_enabled():
            c, d = checkpoint(self._codes, u, vt, w, use_reentrant=False)
        else:
            c, d = self._codes(u, vt, w)
        if g is not None:
            c = c + g
        if self.log_stats:
            self.last = {k: t.detach() for k, t in d.items()}
        delta = self.scaling * F.linear(c.to(x.dtype), self.B.to(x.dtype))
        if self.magnitude is None:
            out = self.base(x) + delta
        else:
            W = self.base.weight
            norm = (W + self.scaling * (self.B @ self.A1).to(W.dtype)).float().norm(dim=1).detach()
            out = (self.magnitude / norm).to(x.dtype) * (F.linear(x, W) + delta)
            if self.base.bias is not None:
                out = out + self.base.bias
        return out.squeeze(0) if squeeze else out


def inject_lena(model, target_modules, **kw):
    """Replace every nn.Linear whose name ends with one of target_modules by a LeNALinear. Returns replaced names."""
    names = []
    for name, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            full = f"{name}.{cname}" if name else cname
            if isinstance(child, nn.Linear) and any(full.endswith(t) for t in target_modules):
                setattr(mod, cname, LeNALinear(child, **kw))
                names.append(full)
    return names


def adapter_state_dict(model):
    """All LeNA tensors except the frozen base layer, keyed by full module name."""
    out = {}
    for name, L in model.named_modules():
        if isinstance(L, LeNALinear):
            for k, v in L.state_dict().items():
                if not k.startswith("base."):
                    out[f"{name}.{k}"] = v.detach().cpu()
    return out


def load_adapter_state_dict(model, sd):
    missing = set(adapter_state_dict(model)) - set(sd)
    assert not missing, f"missing adapter keys: {sorted(missing)[:5]}"
    model.load_state_dict(sd, strict=False)


def trainable_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------------------------------------------
# Baselines in the same code base: LoRA / LoRA-dagger (rank r') and DoRA
# ---------------------------------------------------------------------------------------------------------------
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16, dropout=0.0, dora=False):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        dev = base.weight.device
        self.r, self.scaling = r, alpha / r
        self.A1 = nn.Parameter(torch.empty(r, base.in_features, device=dev))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=dev))
        nn.init.xavier_uniform_(self.A1)
        self.magnitude = nn.Parameter(base.weight.detach().float().norm(dim=1).to(dev)) if dora else None
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        delta = self.scaling * F.linear(F.linear(self.drop(x), self.A1.to(x.dtype)), self.B.to(x.dtype))
        if self.magnitude is None:
            return self.base(x) + delta
        W = self.base.weight
        norm = (W + self.scaling * (self.B @ self.A1).to(W.dtype)).float().norm(dim=1).detach()
        out = (self.magnitude / norm).to(x.dtype) * (F.linear(x, W) + delta)
        return out + self.base.bias if self.base.bias is not None else out


class ANL(nn.Module):
    """Adaptive non-linear layer of AuroRA (Dong et al., NeurIPS 2025; github.com/ins1stenc3/AuroRA, ANL.py),
    copied: tanh(H tanh(z)) + W_s * sum_i B-spline(z_i), cubic B-splines, grid 5 on [-1, 1]."""

    def __init__(self, in_features, out_features, grid_size=5, spline_order=3, scale_noise=0.1, grid_range=(-1, 1)):
        super().__init__()
        self.in_features, self.out_features, self.spline_order = in_features, out_features, spline_order
        h = (grid_range[1] - grid_range[0]) / grid_size
        self.register_buffer("grid", torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0])
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.empty(out_features, grid_size + spline_order))
        nn.init.xavier_normal_(self.base_weight)
        nn.init.uniform_(self.spline_weight, -scale_noise, scale_noise)

    def b_splines(self, x):
        x = x.unsqueeze(-1); g = self.grid
        bases = ((x >= g[:-1]) & (x < g[1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            d1 = g[k:-1] - g[:-k - 1]; d2 = g[k + 1:] - g[1:-k]
            d1 = torch.where(d1 == 0, torch.ones_like(d1), d1); d2 = torch.where(d2 == 0, torch.ones_like(d2), d2)
            bases = (x - g[:-k - 1]) / d1 * bases[..., :-1] + (g[k + 1:] - x) / d2 * bases[..., 1:]
        return bases

    def forward(self, x):
        shp = x.shape; x = x.reshape(-1, self.in_features)
        base = torch.tanh(F.linear(torch.tanh(x), self.base_weight))
        spl = F.linear(self.b_splines(x).sum(dim=1).to(self.spline_weight.dtype), self.spline_weight)
        return (base + spl).reshape(*shp[:-1], self.out_features)


class AuroRALinear(nn.Module):
    """AuroRA baseline: h = W x + (alpha / r) B ANL(A dropout(x)), evaluated with the same dynamic forward
    (AuroRA-D; the released merge path does not run). A: kaiming_uniform(a=sqrt 5) as in the ViT code, B: zeros."""

    def __init__(self, base: nn.Linear, r=2, alpha=4, dropout=0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        dev = base.weight.device
        self.r, self.scaling = r, alpha / r
        self.A1 = nn.Parameter(torch.empty(r, base.in_features, device=dev))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=dev))
        nn.init.kaiming_uniform_(self.A1, a=math.sqrt(5))
        self.anl = ANL(r, r).to(dev)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        z = F.linear(self.drop(x), self.A1.to(x.dtype))
        return self.base(x) + self.scaling * F.linear(self.anl(z.float()).to(x.dtype), self.B.to(x.dtype))


ADAPTERS = {"lena": LeNALinear, "lora": LoRALinear, "aurora": AuroRALinear}
ADAPTERS["lena_aurora"] = lambda base, **kw: LeNALinear(base, base_nl="aurora", **kw)   # LeNA on AuroRA


def inject(model, target_modules, adapter="lena", **kw):
    """Generic injector for LeNA and the LoRA/DoRA baselines."""
    cls = ADAPTERS[adapter]
    names = []
    for name, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            full = f"{name}.{cname}" if name else cname
            if isinstance(child, nn.Linear) and any(full.endswith(t) for t in target_modules):
                setattr(mod, cname, cls(child, **kw))
                names.append(full)
    return names


def adapter_tensors(model):
    """Every non-base tensor of LeNA and LoRA layers (for saving)."""
    out = {}
    for name, L in model.named_modules():
        if isinstance(L, (LeNALinear, LoRALinear, AuroRALinear)):
            for k, v in L.state_dict().items():
                if not k.startswith("base."):
                    out[f"{name}.{k}"] = v.detach().cpu()
    return out


def lora_params(d_pairs, r):
    return sum(r * (di + do) for di, do in d_pairs)


def lena_params(d_pairs, r, K, prior="learned", learn_omega=True):
    per = sum(r * (2 * di + do) for di, do in d_pairs)
    return per + len(d_pairs) * ((r * K if prior != "identity" else 0) + (1 if learn_omega else 0))


def matched_lora_rank(d_pairs, r, K):
    """Smallest LoRA rank r' whose trainable count reaches LeNA's (the LoRA-dagger control)."""
    target = lena_params(d_pairs, r, K)
    rp = r
    while lora_params(d_pairs, rp) < target:
        rp += 1
    return rp
