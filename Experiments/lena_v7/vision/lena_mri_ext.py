"""Opt-in LeNA variants for the MRI sweep. Importing this file changes nothing; `install(model, ...)` patches the
lena_v7 functions of THIS process only, and each patch falls through to the original unless its switch is on.

  ridge      N10 per-knot shrinkage: theta = (n + kappa I)^-1 (m + kappa rho), kappa = softplus(per-module scalar)
  groups     N11 per-scan pooled fit: statistics (n, m, e, ntok) summed over the samples of one group (= scan)
  smooth     N14 smoothness-prior fit: theta = (n + kappa D'D)^-1 (m + kappa D'D rho), D = second difference over knots,
             s = 1 (no Stein step on top); a 1e-4 kappa ridge keeps the system definite when < 2 knots are visited
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import lena_v7 as LV

EXT = {"ridge": False, "smooth": False, "groups": None, "cur": None, "crossfit": False, "neff": False,
       "monotone": False, "deff": None, "rho_tok": None,
       "irls": 0, "huber_c": 1.345, "stream": 0.0, "stream_lam": 0.5, "stream_gen": 0, "acc": None,
       "qmatch": False, "qm_calib": False, "uq": None, "pivot_floor": 0.0, "deff_scale": 1.0}
QLEV = None


def _levels(n, device):
    return (torch.arange(n, device=device, dtype=torch.float32) + 0.5) / n


def interp(x, xp, fp):
    """Batched piecewise-linear interpolation with linear extrapolation. x (..., M); xp, fp (..., Q) increasing xp."""
    Q = xp.shape[-1]
    idx = torch.searchsorted(xp.contiguous(), x.contiguous()).clamp(1, Q - 1)
    x0, x1 = xp.gather(-1, idx - 1), xp.gather(-1, idx)
    y0, y1 = fp.gather(-1, idx - 1), fp.gather(-1, idx)
    t = (x - x0) / (x1 - x0).clamp_min(1e-6)
    return y0 + t * (y1 - y0)


def qmatch_fit(L, rho, mode):
    """R3: theta_k = rho-curve evaluated at T(knot_k), T = monotone map of this input's u-quantiles onto the
    training-set u-quantiles (per channel). F := mean |q_in - q_ref| / sd(q_ref) (shift evidence, for routing)."""
    uq = EXT["uq"]                                                   # (B, r, Q), sorted
    ref = L.uq_ref.to(uq.dtype)[None].expand_as(uq)
    K = rho.shape[-1]
    knots = torch.linspace(L.lo, L.hi, K, device=uq.device, dtype=uq.dtype).expand(*uq.shape[:-1], K)
    Tk = interp(knots, uq, ref)                                      # (B, r, K)
    j, lam = LV.hat_index(Tk, L.lo, L.hi, K)
    rb = rho.expand(*uq.shape[:-1], K)
    theta = rb.gather(-1, j) * (1 - lam) + rb.gather(-1, j + 1) * lam      # rho-curve at T(knot_k)
    Fq = (uq - ref).abs().mean(-1) / ref.std(-1).clamp_min(1e-6)
    z = torch.zeros_like(Fq)
    return theta, dict(s=torch.ones_like(Fq), p=z, F=Fq)
_ORIG = {"fit_and_step": LV.fit_and_step, "pooled_stats": LV.pooled_stats, "spline_eval": LV.spline_eval,
         "prefix_stats": LV.prefix_stats}


def tri_solve(diag, off, rhs):
    """Symmetric tridiagonal solve (Thomas), batched over leading dims. diag, rhs (..., K); off (..., K-1)."""
    K = diag.shape[-1]
    cp, dp = [None] * K, [None] * K
    den = diag[..., 0]
    cp[0] = off[..., 0] / den if K > 1 else None
    dp[0] = rhs[..., 0] / den
    for k in range(1, K):
        den = diag[..., k] - off[..., k - 1] * cp[k - 1]
        if k < K - 1:
            cp[k] = off[..., k] / den
        dp[k] = (rhs[..., k] - off[..., k - 1] * dp[k - 1]) / den
    x = [None] * K
    x[K - 1] = dp[K - 1]
    for k in range(K - 2, -1, -1):
        x[k] = dp[k] - cp[k] * x[k + 1]
    return torch.stack(x, -1)


def _dense(diag, off):
    return torch.diag_embed(diag) + torch.diag_embed(off, 1) + torch.diag_embed(off, -1)


def fit_smooth(diag, off, m, e, ntok, rho, omega, tau, mode):
    K = diag.shape[-1]
    kap = F.softplus(EXT["cur"].kappa_raw).to(m.dtype)
    D = torch.zeros(K - 2, K, device=m.device, dtype=m.dtype)
    idx = torch.arange(K - 2, device=m.device)
    D[idx, idx], D[idx, idx + 1], D[idx, idx + 2] = 1.0, -2.0, 1.0
    P = kap * (D.t() @ D) + 1e-4 * kap * torch.eye(K, device=m.device, dtype=m.dtype)
    A = _dense(diag, off) + P
    rhs = m + (P @ rho.unsqueeze(-1)).squeeze(-1)
    theta = torch.linalg.solve(A, rhs.unsqueeze(-1)).squeeze(-1)
    one = torch.ones_like(ntok)
    return theta, dict(s=one, p=(diag > tau).sum(-1).to(m.dtype), F=torch.zeros_like(ntok),
                       bend=(theta - rho).abs().amax(-1))


def stein_neff(diag, off, m, e, ntok, rho, omega, tau, mode, deff):
    """lena_v7.fit_and_step with the noise variance inflated by the design effect deff = 1 + (N-1) rho_tok."""
    rhs = m - LV.tri_mv(diag, off, rho)
    delta, ret = LV.retained_solve(diag, off, rhs, tau)
    p = ret.sum(-1).to(m.dtype)
    nu = ntok - p
    S = (delta * LV.tri_mv(diag, off, delta)).sum(-1)
    V = (e - 2 * (rho * m).sum(-1) + (rho * LV.tri_mv(diag, off, rho)).sum(-1) - S).clamp_min(0)
    ok = (p >= 3) & (nu >= 4) & (S > 0)
    nu_s = torch.where(ok, nu, torch.full_like(nu, 4.0))
    p_s = torch.where(ok, p, torch.full_like(p, 3.0))
    S_s = torch.where(ok, S, torch.ones_like(S))
    sig2 = torch.maximum(torch.maximum(V / nu_s, 1e-9 * e / ntok.clamp_min(1)), torch.full_like(V, LV.FP32_TINY))
    sig2 = sig2 * deff.to(sig2.dtype)
    gamma = torch.exp(omega) * (p_s - 2) * nu_s / (p_s * (nu_s + 2))
    s = (1 - gamma * p_s * sig2 / S_s).clamp_min(0)
    s = torch.where(ok, s, torch.zeros_like(s))
    Fstat = torch.where(ok, S_s / (p_s * sig2), torch.zeros_like(S))
    return rho + s.unsqueeze(-1) * delta, dict(s=s, p=p, F=Fstat, rho_tok=EXT["rho_tok"].repeat(s.shape[0] // EXT["rho_tok"].shape[0], *([1] * (EXT["rho_tok"].dim() - 1))))


def isotonic(theta):
    """Unweighted isotonic regression along the last dim (max-min formula, exact; differentiable a.e.)."""
    K = theta.shape[-1]
    cs = F.pad(theta.cumsum(-1), (1, 0))
    i = torch.arange(K, device=theta.device)
    cnt = (i[None, :] - i[:, None] + 1).clamp_min(1).to(theta.dtype)          # (i, j) window length
    Mij = (cs[..., None, 1:] - cs[..., :-1, None]) / cnt                       # mean of theta[i..j]
    big = torch.finfo(theta.dtype).max
    valid = i[:, None] <= i[None, :]
    Mij = torch.where(valid, Mij, torch.full_like(Mij, big))
    out = []
    for k in range(K):
        mn = Mij[..., : k + 1, k:].amin(-1)                                      # min over j >= k, for i <= k
        out.append(mn.amax(-1))
    return torch.stack(out, -1)


def fit_and_step(diag, off, m, e, ntok, rho, omega, tau, mode):
    if EXT["qmatch"] and not EXT["qm_calib"] and mode == "lena":
        theta, d = qmatch_fit(EXT["cur"], rho, mode)
    else:
        theta, d = _fit_dispatch(diag, off, m, e, ntok, rho, omega, tau, mode)
    if EXT["acc"] is not None and mode == "lena":                    # running means of s and curve bend
        a = EXT["acc"]
        a["s"] = a.get("s", 0.0) + d["s"].float().mean().item()
        a["bend"] = a.get("bend", 0.0) + (theta - rho).abs().amax(-1).float().mean().item()
        a["n"] = a.get("n", 0) + 1
    if EXT.get("acc_eval") is not None and mode == "lena" and not torch.is_grad_enabled():   # eval-only running means
        a = EXT["acc_eval"]
        a["s"] = a.get("s", 0.0) + d["s"].float().mean().item()
        a["bend"] = a.get("bend", 0.0) + (theta - rho).abs().amax(-1).float().mean().item()
        a["n"] = a.get("n", 0) + 1
    return theta, d


def _fit_dispatch(diag, off, m, e, ntok, rho, omega, tau, mode):
    if EXT["pivot_floor"] > 0:                                       # H11: ridge lambda_min tr(n)/K on the diagonal
        diag = diag + EXT["pivot_floor"] * diag.sum(-1, keepdim=True) / diag.shape[-1]
    if EXT["neff"] and mode == "lena" and EXT["deff"] is not None:
        theta, d = stein_neff(diag, off, m, e, ntok, rho, omega, tau, mode, EXT["deff"])
        return (isotonic(theta) if EXT["monotone"] else theta), d
    if EXT["monotone"]:
        theta, d = _ORIG["fit_and_step"](diag, off, m, e, ntok, rho, omega, tau, mode)
        return isotonic(theta), d
    if EXT["smooth"] and mode == "lena":
        return fit_smooth(diag, off, m, e, ntok, rho, omega, tau, mode)
    if not EXT["ridge"] or mode != "lena":
        return _ORIG["fit_and_step"](diag, off, m, e, ntok, rho, omega, tau, mode)
    kap = F.softplus(EXT["cur"].kappa_raw).to(m.dtype)
    theta = tri_solve(diag + kap, off, m + kap * rho)
    shrink = (diag / (diag + kap)).mean(-1)                         # mean per-knot weight on the fit (diagnostic)
    return theta, dict(s=shrink, p=(diag > tau).sum(-1).to(m.dtype), F=torch.zeros_like(ntok))


def checker(N, device):
    """(N,) bool, True on the 'A' squares of a checkerboard over the sqrt(N) x sqrt(N) token grid."""
    g = int(round(N ** 0.5))
    assert g * g == N, f"crossfit needs a square token grid, got N={N}"
    i = torch.arange(g, device=device)
    return ((i[:, None] + i[None, :]) % 2 == 0).flatten()


def lag1_corr(u):
    """Mean lag-1 spatial autocorrelation of u (B, N, r) over the token grid, per input and channel (detached)."""
    B, N, r = u.shape
    g = int(round(N ** 0.5))
    if g * g != N:
        g1 = int(round((N - 1) ** 0.5))
        if N > 1 and g1 * g1 == N - 1:                              # ViT with a leading CLS token: grid = tokens 1..N-1
            u, N, g = u[:, 1:], N - 1, g1
        else:
            return torch.zeros(B, r, device=u.device, dtype=u.dtype)
    x = u.detach().reshape(B, g, g, r)
    x = x - x.mean((1, 2), keepdim=True)
    var = (x * x).mean((1, 2)).clamp_min(1e-12)
    ch = (x[:, :, 1:] * x[:, :, :-1]).mean((1, 2)) / var
    cv = (x[:, 1:] * x[:, :-1]).mean((1, 2)) / var
    return (0.5 * (ch + cv)).clamp(0, 0.999)


def prefix_lag1(u, w, L=None):
    """Causal R1: lag-1 autocorrelation of u over the prefix <= t (the window of lena_v7.prefix_stats, so no future
    token enters), per input, position and channel; detached. Token weights w (padding = 0) weight the moments and
    the pairs (w_t w_{t-1}). In stream decoding (lena_stream, eval) the running sums and the last token are carried
    on the layer across calls and reset whenever lena_stream resets the layer state. Returns (rho_t, n_t)."""
    x = u.detach().float()
    ww = w.detach().float().expand_as(x)
    xp = F.pad(x, (0, 0, 1, 0))[:, :-1]
    wp = ww * F.pad(ww, (0, 0, 1, 0))[:, :-1]
    cs = [t.cumsum(1) for t in (ww, ww * x, ww * x * x, wp, wp * x * xp)]
    stream = L is not None and LV._CTX.stream and not L.training
    if stream:
        st = getattr(L, "_neff_state", None) if L._state is not None else None
        if st is not None:
            lx, lw, sums = st
            bw = lw * ww[:, :1]                                        # pair across the call boundary
            cs[3] = cs[3] + bw
            cs[4] = cs[4] + bw * lx * x[:, :1]
            cs = [c + q.unsqueeze(1) for c, q in zip(cs, sums)]
        L._neff_state = (x[:, -1:], ww[:, -1:], [c[:, -1] for c in cs])
    n, s1, s2, pn, pp = cs
    mu = s1 / n.clamp_min(1e-6)
    var = s2 / n.clamp_min(1e-6) - mu * mu
    cov = pp / pn.clamp_min(1e-6) - mu * mu
    ok = (pn >= 1) & (var > 1e-12)
    rho = torch.where(ok, cov / var.clamp_min(1e-12), torch.zeros_like(var)).clamp(0, 0.999)
    return rho.to(u.dtype), n.to(u.dtype)


def prefix_stats(u, v, w, lo, hi, K):
    if EXT["neff"]:                                                 # R1, causal: design effect of the prefix <= t
        rt, n = prefix_lag1(u, w, EXT["cur"])
        EXT["rho_tok"], EXT["deff"] = rt, 1 + EXT["deff_scale"] * (n - 1).clamp_min(0) * rt
    return _ORIG["prefix_stats"](u, v, w, lo, hi, K)


def _mode_of(L):
    return LV._CTX.force_mode or L.mode


def irls_weights(u, v, w, lo, hi, K, L):
    """R11: K rounds of Huber reweighting of the tokens of each input/channel around the current per-input fit."""
    rho, omega = L.rho.to(u.dtype), L.omega.to(u.dtype)
    wc = w
    for _ in range(EXT["irls"]):
        j, lam, diag, off, m, e, nt = _ORIG["pooled_stats"](u, v, wc, lo, hi, K)
        theta, _ = _ORIG["fit_and_step"](diag, off, m, e, nt, rho, omega, L.tau, _mode_of(L))
        res = v - _ORIG["spline_eval"](theta.unsqueeze(1), j, lam)                  # (B, N, r)
        med = res.median(1, keepdim=True).values
        sig = (1.4826 * (res - med).abs().median(1, keepdim=True).values).clamp_min(1e-6)
        z = (res / sig).abs()
        hub = torch.where(z <= EXT["huber_c"], torch.ones_like(z), EXT["huber_c"] / z.clamp_min(1e-12))
        wc = w * hub
    return wc


def stream_mix(L, acc):
    """R10: mix the input's statistics with an EMA of the statistics seen so far in this evaluation stream."""
    if getattr(L, "_ema_gen", None) != EXT["stream_gen"]:
        L._ema, L._ema_gen = None, EXT["stream_gen"]
    cur = [a.detach().float().mean(0, keepdim=True) for a in acc]
    prev = L._ema if L._ema is not None else cur
    lam, beta = EXT["stream_lam"], EXT["stream"]
    mixed = [((1 - lam) * a.float() + lam * p_.expand_as(a)).to(a.dtype) for a, p_ in zip(acc, prev)]
    L._ema = [beta * p_ + (1 - beta) * c for p_, c in zip(prev, cur)]
    return mixed


def pooled_stats(u, v, w, lo, hi, K):
    L = EXT["cur"]
    if EXT["qmatch"] and L is not None:
        q = torch.quantile(u.detach().float(), _levels(16, u.device), dim=1).permute(1, 2, 0)   # (B, r, 16)
        q = q.cummax(-1).values + 1e-6 * torch.arange(16, device=u.device)                 # strictly increasing
        EXT["uq"] = q.to(u.dtype)
        if EXT["qm_calib"]:
            L.uq_sum = getattr(L, "uq_sum", 0) + q.mean(0); L.uq_n = getattr(L, "uq_n", 0) + 1
    if (EXT["irls"] or EXT["stream"]) and L is not None and _mode_of(L) == "lena":
        if EXT["irls"]:
            w = irls_weights(u, v, w, lo, hi, K, L)
        j, lam, *acc = _ORIG["pooled_stats"](u, v, w, lo, hi, K)
        if EXT["stream"] and not L.training:
            acc = stream_mix(L, acc)
        return (j, lam, *acc)
    if EXT["neff"]:                                                 # R1: design effect of correlated tokens
        rt = lag1_corr(u)
        N = u.shape[1]
        EXT["rho_tok"], EXT["deff"] = rt, 1 + EXT["deff_scale"] * (N - 1) * rt     # deff_scale: eval-time shrink knob (1 = R1)
    if EXT["crossfit"]:                                             # R5: stats from one half, applied to the other
        a = checker(u.shape[1], u.device).to(w.dtype)[None, :, None]
        j, lam, *sa = _ORIG["pooled_stats"](u, v, w * a, lo, hi, K)
        _, _, *sb = _ORIG["pooled_stats"](u, v, w * (1 - a), lo, hi, K)
        if EXT["deff"] is not None:
            EXT["deff"] = torch.cat([EXT["deff"], EXT["deff"]])
        return (j, lam, *[torch.cat([x, y]) for x, y in zip(sa, sb)])   # [fit on A ; fit on B]
    j, lam, *acc = _ORIG["pooled_stats"](u, v, w, lo, hi, K)
    g = EXT["groups"]
    if g is None:
        return (j, lam, *acc)
    B = acc[0].shape[0]
    assert B == g.numel(), f"group ids {g.numel()} != batch {B} (one SAM window per slice expected at 224)"
    oh = F.one_hot(g, int(g.max()) + 1).to(acc[0].dtype).t()       # (G, B)
    pooled = [(oh @ a.reshape(B, -1)).reshape(-1, *a.shape[1:])[g] for a in acc]
    return (j, lam, *pooled)


def spline_eval(theta, j, lam):
    if EXT["crossfit"] and theta.shape[0] == 2 * j.shape[0]:
        B = j.shape[0]
        a = checker(j.shape[1], j.device)[None, :, None]
        from_a = _ORIG["spline_eval"](theta[:B], j, lam)            # curve fitted on A ...
        from_b = _ORIG["spline_eval"](theta[B:], j, lam)
        return torch.where(a, from_b, from_a)                       # ... applied to B tokens, and vice versa
    return _ORIG["spline_eval"](theta, j, lam)


def make_monotone(L):
    """R6: rho = rho_1 + cumsum(softplus(delta)) (replaces the rho parameter of one LeNA layer)."""
    rho0 = L._parameters.pop("rho").detach()
    L.register_parameter("rho_first", nn.Parameter(rho0[:, :1].clone()))
    gap = (rho0[:, 1:] - rho0[:, :-1]).clamp_min(1e-3)
    L.register_parameter("rho_delta", nn.Parameter(torch.log(torch.expm1(gap))))
    L.register_forward_pre_hook(lambda mod, args: setattr(
        mod, "rho", torch.cat([mod.rho_first, mod.rho_first + F.softplus(mod.rho_delta).cumsum(-1)], -1)))
    L.rho = torch.cat([L.rho_first, L.rho_first + F.softplus(L.rho_delta).cumsum(-1)], -1).detach()


def qmatch_finalize(model):
    """End of the calibration pass: reference quantiles = mean of the batch-mean quantiles, per layer/channel."""
    for L in LV.lena_layers(model):
        L.register_buffer("uq_ref", (L.uq_sum / L.uq_n).detach())


def triton_neff_ok():
    """True when n_eff (LeNA-S) is the ONLY active patch, so lena_v7._codes may use the fused Triton kernel with deff.
    Any other switch, or a foreign re-patch of the lena_v7 functions, keeps the PyTorch path."""
    E = EXT
    return bool(E["neff"] and not (E["ridge"] or E["smooth"] or E["crossfit"] or E["monotone"] or E["qmatch"]
                                   or E["qm_calib"] or E["irls"] or E["stream"] or E["pivot_floor"])
                and E["groups"] is None and LV.fit_and_step is fit_and_step and LV.pooled_stats is pooled_stats
                and LV.spline_eval is spline_eval and LV.prefix_stats is prefix_stats)


def triton_neff_codes(L, fused_codes, u, v, w, rho, omega, causal):
    """LeNA-S on the fused Triton path: deff exactly as pooled_stats / prefix_stats compute it (detached), the fit of
    stein_neff inside the kernel, and the same diagnostics (rho_tok, acc / acc_eval running means) as fit_and_step."""
    EXT["cur"] = L
    if causal:
        rt, n = prefix_lag1(u, w, L)
        deff = 1 + EXT["deff_scale"] * (n - 1).clamp_min(0) * rt
    else:
        rt = lag1_corr(u)
        N = u.shape[1]
        deff = 1 + EXT["deff_scale"] * (N - 1) * rt
    EXT["rho_tok"], EXT["deff"] = rt, deff
    c, d = fused_codes(u, v, w, rho, omega, L.lo, L.hi, L.tau, causal, deff=deff)
    bend = d.pop("bend")
    d["rho_tok"] = rt
    for a in (EXT["acc"], EXT.get("acc_eval") if not torch.is_grad_enabled() else None):
        if a is not None:
            a["s"] = a.get("s", 0.0) + d["s"].float().mean().item()
            a["bend"] = a.get("bend", 0.0) + bend.float().mean().item()
            a["n"] = a.get("n", 0) + 1
    return c, d


def install_global(irls=0, stream=0.0, stream_lam=0.5, acc=True):
    """Eval-time patches without a model handle: a global module pre-hook tracks the LeNA layer being run."""
    LV.fit_and_step, LV.pooled_stats, LV.spline_eval = fit_and_step, pooled_stats, spline_eval
    # the fused Triton path bypasses these patched PyTorch functions (n_eff etc. would be silently ignored)
    if LV._TRITON["on"]:
        print("lena_mri_ext: fused Triton path disabled for patched LeNA layers (PyTorch fit path)", flush=True)
    LV._TRITON["patched"] = True
    EXT["irls"], EXT["stream"], EXT["stream_lam"] = irls, stream, stream_lam
    EXT["acc"] = {} if acc else None
    nn.modules.module.register_module_forward_pre_hook(
        lambda mod, args: EXT.__setitem__("cur", mod) if isinstance(mod, LV.LeNALinear) else None)


def install(model, ridge=False, kappa0=4.0, smooth=False, crossfit=False, neff=False, monotone=False, irls=0,
            pivot_floor=0.0):
    LV.fit_and_step, LV.pooled_stats, LV.spline_eval = fit_and_step, pooled_stats, spline_eval
    # the fused Triton path bypasses these patched PyTorch functions; only n_eff alone is implemented in the kernel
    # (lena_triton NEFF, gated per call by triton_neff_ok); every other patch keeps the PyTorch fit path
    if LV._TRITON["on"]:
        alone = neff and not (ridge or smooth or crossfit or monotone or irls or pivot_floor)
        print("lena_mri_ext: " + ("n_eff on the fused Triton path (deff in the kernel)" if alone else
                                  "fused Triton path disabled for patched LeNA layers (PyTorch fit path)"), flush=True)
    LV._TRITON["patched"] = True
    LV._TRITON["neff_ok"], LV._TRITON["neff_codes"] = triton_neff_ok, triton_neff_codes   # n_eff alone: fused path
    EXT["ridge"], EXT["smooth"], EXT["crossfit"], EXT["neff"], EXT["monotone"] = ridge, smooth, crossfit, neff, monotone
    EXT["irls"], EXT["pivot_floor"] = irls, pivot_floor
    if irls or neff:
        for L in LV.lena_layers(model):
            L.register_forward_pre_hook(lambda mod, args: EXT.__setitem__("cur", mod))
    if neff:
        LV.prefix_stats = prefix_stats
        if not getattr(LV.LeNALinear, "_neff_reorder", False):         # beam search: reorder the n_eff stream state too
            _orig_reorder = LV.LeNALinear.reorder_state

            def reorder_state(self, idx):
                _orig_reorder(self, idx)
                st = getattr(self, "_neff_state", None)
                if st is not None:
                    sel = lambda t: t.index_select(0, idx.to(t.device))
                    self._neff_state = (sel(st[0]), sel(st[1]), [sel(c) for c in st[2]])
            LV.LeNALinear.reorder_state = reorder_state
            LV.LeNALinear._neff_reorder = True
    if monotone:
        for L in LV.lena_layers(model):
            make_monotone(L)
    if ridge or smooth:
        raw = math.log(math.expm1(kappa0))
        for L in LV.lena_layers(model):
            L.register_parameter("kappa_raw", nn.Parameter(torch.tensor(raw, device=L.A1.device)))
            L.register_forward_pre_hook(lambda mod, args: EXT.__setitem__("cur", mod))


class groups:
    """Context: per-sample group ids (B,) for the pooled fit."""

    def __init__(self, g):
        self.g = g

    def __enter__(self):
        self.old, EXT["groups"] = EXT["groups"], self.g

    def __exit__(self, *a):
        EXT["groups"] = self.old
