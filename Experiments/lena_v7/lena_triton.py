"""Fused Triton path of LeNALinear._codes for mode "lena" (lena_v7.set_triton(True)): statistics, retained-knot solve,
James-Stein step and curve evaluation in one kernel, and a backward kernel with the hand-derived gradients w.r.t. u, v,
rho and omega (retained set piecewise constant, as autograd on the PyTorch path).

One program per (input b, channel k); tokens are the vector lanes, processed in chunks of BLOCK. Causal mode fits every
lane on its prefix statistics (chunk cumsum + carry); pooled mode fits once on the totals. Per-lane K-vectors live in
(BLOCK, KP) register tiles; the sequential LDL^T sweeps read and write one knot column at a time.

NEFF (LeNA-S, vision/lena_mri_ext.py stein_neff): the noise variance is inflated by a per-lane design effect,
sig2 = max(V/nu, 1e-9 E/T, tiny) * deff, with deff (B, r) pooled or (B, N, r) causal computed outside from the
detached token autocorrelation, so deff is a constant of the backward (as on the PyTorch path). With NEFF the kernel
also writes bend = max_k |s delta_k| for the lena_mri_ext running diagnostics. NEFF=False compiles the original code.
"""
import numpy as np
import torch
import triton
import triton.language as tl

TINY = tl.constexpr(1.1754943508222875e-38)      # torch.finfo(torch.float32).tiny


@triton.jit
def _col(X, kk, k):
    return tl.sum(tl.where(kk[None, :] == k, X, 0.0), 1)


@triton.jit
def _put(X, kk, k, x):
    return tl.where(kk[None, :] == k, x[:, None], X)


@triton.jit
def _token(U, V, W, b, k, offs, N, R, swb, swn, swk, lo, inv_h, K: tl.constexpr):
    msk = offs < N
    idx = b * N * R + offs * R + k
    u = tl.load(U + idx, mask=msk, other=0.0)
    v = tl.load(V + idx, mask=msk, other=0.0)
    w = tl.load(W + b * swb + offs * swn + k * swk, mask=msk, other=0.0)
    z = (u - lo) * inv_h
    jf = tl.minimum(tl.maximum(tl.floor(z), 0.0), K - 2.0)
    return idx, msk, v, w, jf.to(tl.int32), z - jf


@triton.jit
def _contrib(j, lam, v, w, kk):
    oh = kk[None, :] == j[:, None]
    oh1 = kk[None, :] == (j + 1)[:, None]
    a = (1 - lam) * w
    b = lam * w
    dD = tl.where(oh, (a * (1 - lam))[:, None], 0.0) + tl.where(oh1, (b * lam)[:, None], 0.0)
    dO = tl.where(oh, (a * lam)[:, None], 0.0)                       # n_{j,j+1} at slot j
    dOP = tl.where(oh1, (a * lam)[:, None], 0.0)                     # same, shifted to slot j+1
    dM = tl.where(oh, (a * v)[:, None], 0.0) + tl.where(oh1, (b * v)[:, None], 0.0)
    return oh, oh1, dD, dO, dOP, dM, w * v * v


@triton.jit
def _fit_fwd(D, O, OP, M, E, T, rho, rhon, rhop, eo, tau, kk, K: tl.constexpr, deff, NEFF: tl.constexpr):
    zero = D * 0.0
    z1 = E * 0.0
    NRHO = D * rho[None, :] + O * rhon[None, :] + OP * rhop[None, :]
    RHS = M - NRHO
    PIV = zero
    Z = zero
    LL = zero
    RET = zero
    pd = z1
    pz = z1
    pr = z1 > 1.0
    for k in tl.static_range(K):
        Dk = _col(D, kk, k)
        rhs = _col(RHS, kk, k)
        if k == 0:
            d = Dk
            z = rhs
            l = z1
        else:
            c = tl.where(pr, _col(OP, kk, k), 0.0)
            l = c / pd
            d = Dk - l * c
            z = rhs - l * pz
        r = d > tau
        d = tl.where(r, d, 1.0)
        z = tl.where(r, z, 0.0)
        PIV = _put(PIV, kk, k, d)
        Z = _put(Z, kk, k, z)
        LL = _put(LL, kk, k, tl.where(r, l, 0.0))
        RET = _put(RET, kk, k, r.to(tl.float32))
        pd = d
        pz = z
        pr = r
    DELTA = zero
    DNEXT = zero
    DPREV = zero
    dn = z1
    for kr in tl.static_range(K):
        q = K - 1 - kr
        rk = _col(RET, kk, q) > 0.5
        if q == K - 1:
            dk = _col(Z, kk, q) / _col(PIV, kk, q)
        else:
            dk = _col(Z, kk, q) / _col(PIV, kk, q) - _col(LL, kk, q + 1) * dn
        dk = tl.where(rk, dk, 0.0)
        DELTA = _put(DELTA, kk, q, dk)
        DNEXT = _put(DNEXT, kk, q, dn)
        if q + 1 < K:
            DPREV = _put(DPREV, kk, q + 1, dk)
        dn = dk
    p = tl.sum(RET, 1)
    nu = T - p
    S = tl.sum(DELTA * (D * DELTA + O * DNEXT + OP * DPREV), 1)
    Vp = E - 2 * tl.sum(rho[None, :] * M, 1) + tl.sum(rho[None, :] * NRHO, 1) - S
    V = tl.maximum(Vp, 0.0)
    ok = (p >= 3) & (nu >= 4) & (S > 0)
    nus = tl.where(ok, nu, 4.0)
    ps = tl.where(ok, p, 3.0)
    Ss = tl.where(ok, S, 1.0)
    Tc = tl.maximum(T, 1.0)
    sa = V / nus
    sb = 1e-9 * E / Tc
    m1 = tl.maximum(sa, sb)
    sig2 = tl.maximum(m1, TINY)
    if NEFF:
        sig2 = sig2 * deff                                            # design effect (constant of the backward)
    gam = eo * (ps - 2) * nus / (ps * (nus + 2))
    sp = 1 - gam * ps * sig2 / Ss
    s = tl.where(ok, tl.maximum(sp, 0.0), 0.0)
    Fs = tl.where(ok, Ss / (ps * sig2), 0.0)
    return NRHO, PIV, LL, RET, DELTA, DNEXT, DPREV, p, s, Fs, ok, Vp, nus, ps, Ss, Tc, sa, sb, m1, sig2, gam, sp


@triton.jit
def _fit_bwd(GT, D, O, OP, M, rho, rhon, NRHO, PIV, LL, RET, DELTA, DNEXT, DPREV, s, ok, Vp, nus, ps, Ss, Tc,
             sa, sb, m1, sig2, gam, sp, kk, K: tl.constexpr, deff, NEFF: tl.constexpr):
    """Per-lane gradients of the fit given gtheta: (gD, gO, gM) tiles, gE, grho tile, gomega."""
    zero = D * 0.0
    z1 = s * 0.0
    gs = tl.where(ok & (sp >= 0), tl.sum(GT * DELTA, 1), 0.0)
    ggam = -gs * ps * sig2 / Ss
    gsig2 = -gs * gam * ps / Ss
    gS = gs * gam * ps * sig2 / (Ss * Ss)
    if NEFF:
        gsig2 = gsig2 * deff                                          # d(m1c * deff)/d m1c
    gm1 = tl.where(m1 > TINY, gsig2, tl.where(m1 == TINY, 0.5 * gsig2, 0.0))   # torch.maximum splits ties
    ga = tl.where(sa > sb, gm1, tl.where(sa == sb, 0.5 * gm1, 0.0))
    gV = tl.where(Vp >= 0, ga / nus, 0.0)
    gE = gV + (gm1 - ga) * 1e-9 / Tc
    gS = gS - gV
    G = RET * (s[:, None] * GT + 2 * gS[:, None] * (D * DELTA + O * DNEXT + OP * DPREV))
    Y = zero
    yp = z1
    for k in tl.static_range(K):                                   # adjoint solve with the same LDL^T factors
        if k == 0:
            y = _col(G, kk, k)
        else:
            y = _col(G, kk, k) - _col(LL, kk, k) * yp
        y = tl.where(_col(RET, kk, k) > 0.5, y, 0.0)
        Y = _put(Y, kk, k, y)
        yp = y
    X = zero
    XN = zero
    XP = zero
    xn = z1
    for kr in tl.static_range(K):
        q = K - 1 - kr
        if q == K - 1:
            x = _col(Y, kk, q) / _col(PIV, kk, q)
        else:
            x = _col(Y, kk, q) / _col(PIV, kk, q) - _col(LL, kk, q + 1) * xn
        x = tl.where(_col(RET, kk, q) > 0.5, x, 0.0)
        X = _put(X, kk, q, x)
        XN = _put(XN, kk, q, xn)
        if q + 1 < K:
            XP = _put(XP, kk, q + 1, x)
        xn = x
    A = rho[None, :] + DELTA
    AN = rhon[None, :] + DNEXT
    gVb = gV[:, None]
    gSb = gS[:, None]
    GD = -X * A + gVb * rho[None, :] * rho[None, :] + gSb * DELTA * DELTA
    GO = -(X * AN + XN * A) + 2 * gVb * rho[None, :] * rhon[None, :] + 2 * gSb * DELTA * DNEXT
    GM = X - 2 * gVb * rho[None, :]
    GR = GT - (D * X + O * XN + OP * XP) - 2 * gVb * M + 2 * gVb * NRHO
    return GD, GO, GM, gE, GR, ggam * gam


@triton.jit
def _load_rho(RHO, EO, k, kk, K: tl.constexpr):
    base = RHO + k * K
    rho = tl.load(base + kk, mask=kk < K, other=0.0)
    rhon = tl.load(base + kk + 1, mask=kk + 1 < K, other=0.0)
    rhop = tl.load(base + kk - 1, mask=(kk >= 1) & (kk < K), other=0.0)
    return rho, rhon, rhop, tl.load(EO)


@triton.jit
def _fwd_kernel(U, V, W, RHO, EO, C, SO, PO, FO, CAR, DEFF, BO, N, R, NCH, swb, swn, swk, lo, inv_h, tau,
                K: tl.constexpr, KP: tl.constexpr, BLOCK: tl.constexpr, CAUSAL: tl.constexpr, NEFF: tl.constexpr):
    pid = tl.program_id(0)
    b = pid // R
    k = pid % R
    kk = tl.arange(0, KP)
    ln = tl.arange(0, BLOCK)
    rho, rhon, rhop, eo = _load_rho(RHO, EO, k, kk, K)
    cD = tl.zeros([KP], tl.float32)
    cO = cD
    cOP = cD
    cM = cD
    cE = tl.sum(cD, 0)
    cT = cE
    if CAUSAL:
        for ch in range(0, NCH):
            idx, msk, v, w, j, lam = _token(U, V, W, b, k, ch * BLOCK + ln, N, R, swb, swn, swk, lo, inv_h, K)
            oh, oh1, dD, dO, dOP, dM, dE = _contrib(j, lam, v, w, kk)
            cb = CAR + (pid * NCH + ch) * 5 * KP
            tl.store(cb + kk, cD)
            tl.store(cb + KP + kk, cO)
            tl.store(cb + 2 * KP + kk, cOP)
            tl.store(cb + 3 * KP + kk, cM)
            tl.store(cb + 4 * KP + kk, tl.where(kk == 0, cE, tl.where(kk == 1, cT, 0.0)))
            D = tl.cumsum(dD, 0) + cD[None, :]
            O = tl.cumsum(dO, 0) + cO[None, :]
            OP = tl.cumsum(dOP, 0) + cOP[None, :]
            M = tl.cumsum(dM, 0) + cM[None, :]
            E = tl.cumsum(dE, 0) + cE
            T = tl.cumsum(w, 0) + cT
            if NEFF:
                dl = tl.load(DEFF + idx, mask=msk, other=1.0)
            else:
                dl = w
            cD += tl.sum(dD, 0)
            cO += tl.sum(dO, 0)
            cOP += tl.sum(dOP, 0)
            cM += tl.sum(dM, 0)
            cE += tl.sum(dE, 0)
            cT += tl.sum(w, 0)
            NRHO, PIV, LL, RET, DELTA, DNEXT, DPREV, p, s, Fs, ok, Vp, nus, ps, Ss, Tc, sa, sb, m1, sig2, gam, sp = \
                _fit_fwd(D, O, OP, M, E, T, rho, rhon, rhop, eo, tau, kk, K, dl, NEFF)
            TH = rho[None, :] + s[:, None] * DELTA
            if NEFF:
                tl.store(BO + idx, tl.max(tl.abs(s[:, None] * DELTA), 1), mask=msk)
            c = tl.sum(tl.where(oh, TH, 0.0), 1) * (1 - lam) + tl.sum(tl.where(oh1, TH, 0.0), 1) * lam
            tl.store(C + idx, c, mask=msk)
            tl.store(SO + idx, s, mask=msk)
            tl.store(PO + idx, p, mask=msk)
            tl.store(FO + idx, Fs, mask=msk)
    else:
        for ch in range(0, NCH):
            idx, msk, v, w, j, lam = _token(U, V, W, b, k, ch * BLOCK + ln, N, R, swb, swn, swk, lo, inv_h, K)
            oh, oh1, dD, dO, dOP, dM, dE = _contrib(j, lam, v, w, kk)
            cD += tl.sum(dD, 0)
            cO += tl.sum(dO, 0)
            cOP += tl.sum(dOP, 0)
            cM += tl.sum(dM, 0)
            cE += tl.sum(dE, 0)
            cT += tl.sum(w, 0)
        z2 = tl.zeros([BLOCK, KP], tl.float32)
        z1 = tl.zeros([BLOCK], tl.float32)
        if NEFF:
            dl = z1 + tl.load(DEFF + pid)
        else:
            dl = z1
        NRHO, PIV, LL, RET, DELTA, DNEXT, DPREV, p, s, Fs, ok, Vp, nus, ps, Ss, Tc, sa, sb, m1, sig2, gam, sp = \
            _fit_fwd(z2 + cD[None, :], z2 + cO[None, :], z2 + cOP[None, :], z2 + cM[None, :], z1 + cE, z1 + cT,
                     rho, rhon, rhop, eo, tau, kk, K, dl, NEFF)
        first = ln == 0
        if NEFF:
            tl.store(BO + pid + ln, tl.max(tl.abs(s[:, None] * DELTA), 1), mask=first)
        th = tl.sum(tl.where(first[:, None], rho[None, :] + s[:, None] * DELTA, 0.0), 0)
        tl.store(SO + pid + ln, s, mask=first)
        tl.store(PO + pid + ln, p, mask=first)
        tl.store(FO + pid + ln, Fs, mask=first)
        for ch in range(0, NCH):
            idx, msk, v, w, j, lam = _token(U, V, W, b, k, ch * BLOCK + ln, N, R, swb, swn, swk, lo, inv_h, K)
            oh = kk[None, :] == j[:, None]
            oh1 = kk[None, :] == (j + 1)[:, None]
            c = tl.sum(tl.where(oh, th[None, :], 0.0), 1) * (1 - lam) + tl.sum(tl.where(oh1, th[None, :], 0.0), 1) * lam
            tl.store(C + idx, c, mask=msk)


@triton.jit
def _token_grads(GC, DU, DV, idx, msk, v, w, lam, oh, oh1, TH, RD, RO, RM, RE, dc, inv_h):
    tDj = tl.sum(tl.where(oh, RD, 0.0), 1)
    tDj1 = tl.sum(tl.where(oh1, RD, 0.0), 1)
    tOj = tl.sum(tl.where(oh, RO, 0.0), 1)
    tMj = tl.sum(tl.where(oh, RM, 0.0), 1)
    tMj1 = tl.sum(tl.where(oh1, RM, 0.0), 1)
    thj = tl.sum(tl.where(oh, TH, 0.0), 1)
    thj1 = tl.sum(tl.where(oh1, TH, 0.0), 1)
    dlam = w * (-2 * (1 - lam) * tDj + 2 * lam * tDj1 + (1 - 2 * lam) * tOj + v * (tMj1 - tMj)) + dc * (thj1 - thj)
    tl.store(DU + idx, dlam * inv_h, mask=msk)
    tl.store(DV + idx, w * ((1 - lam) * tMj + lam * tMj1 + 2 * v * RE), mask=msk)


@triton.jit
def _bwd_kernel(U, V, W, RHO, EO, GC, CAR, DEFF, DU, DV, GRHO, GOM, N, R, NCH, swb, swn, swk, lo, inv_h, tau,
                K: tl.constexpr, KP: tl.constexpr, BLOCK: tl.constexpr, CAUSAL: tl.constexpr, NEFF: tl.constexpr):
    pid = tl.program_id(0)
    b = pid // R
    k = pid % R
    kk = tl.arange(0, KP)
    ln = tl.arange(0, BLOCK)
    rho, rhon, rhop, eo = _load_rho(RHO, EO, k, kk, K)
    gr = tl.zeros([KP], tl.float32)
    gom = tl.sum(gr, 0)
    if CAUSAL:
        gcD = gr
        gcO = gr
        gcM = gr
        gcE = gom
        for cr in range(0, NCH):
            ch = NCH - 1 - cr
            idx, msk, v, w, j, lam = _token(U, V, W, b, k, ch * BLOCK + ln, N, R, swb, swn, swk, lo, inv_h, K)
            oh, oh1, dD, dO, dOP, dM, dE = _contrib(j, lam, v, w, kk)
            cb = CAR + (pid * NCH + ch) * 5 * KP
            cET = tl.load(cb + 4 * KP + kk)
            D = tl.cumsum(dD, 0) + tl.load(cb + kk)[None, :]
            O = tl.cumsum(dO, 0) + tl.load(cb + KP + kk)[None, :]
            OP = tl.cumsum(dOP, 0) + tl.load(cb + 2 * KP + kk)[None, :]
            M = tl.cumsum(dM, 0) + tl.load(cb + 3 * KP + kk)[None, :]
            E = tl.cumsum(dE, 0) + tl.sum(tl.where(kk == 0, cET, 0.0), 0)
            T = tl.cumsum(w, 0) + tl.sum(tl.where(kk == 1, cET, 0.0), 0)
            if NEFF:
                dl = tl.load(DEFF + idx, mask=msk, other=1.0)
            else:
                dl = w
            NRHO, PIV, LL, RET, DELTA, DNEXT, DPREV, p, s, Fs, ok, Vp, nus, ps, Ss, Tc, sa, sb, m1, sig2, gam, sp = \
                _fit_fwd(D, O, OP, M, E, T, rho, rhon, rhop, eo, tau, kk, K, dl, NEFF)
            dc = tl.load(GC + idx, mask=msk, other=0.0)
            GT = tl.where(oh, (dc * (1 - lam))[:, None], 0.0) + tl.where(oh1, (dc * lam)[:, None], 0.0)
            GD, GO, GM, gE, GR, go = _fit_bwd(GT, D, O, OP, M, rho, rhon, NRHO, PIV, LL, RET, DELTA, DNEXT, DPREV, s,
                                              ok, Vp, nus, ps, Ss, Tc, sa, sb, m1, sig2, gam, sp, kk, K, dl, NEFF)
            gr += tl.sum(GR, 0)
            gom += tl.sum(go, 0)
            RD = tl.cumsum(GD, 0, reverse=True) + gcD[None, :]
            RO = tl.cumsum(GO, 0, reverse=True) + gcO[None, :]
            RM = tl.cumsum(GM, 0, reverse=True) + gcM[None, :]
            RE = tl.cumsum(gE, 0, reverse=True) + gcE
            gcD += tl.sum(GD, 0)
            gcO += tl.sum(GO, 0)
            gcM += tl.sum(GM, 0)
            gcE += tl.sum(gE, 0)
            TH = rho[None, :] + s[:, None] * DELTA
            _token_grads(GC, DU, DV, idx, msk, v, w, lam, oh, oh1, TH, RD, RO, RM, RE, dc, inv_h)
    else:
        cD = gr
        cO = gr
        cOP = gr
        cM = gr
        cE = gom
        cT = gom
        gt = gr
        for ch in range(0, NCH):
            idx, msk, v, w, j, lam = _token(U, V, W, b, k, ch * BLOCK + ln, N, R, swb, swn, swk, lo, inv_h, K)
            oh, oh1, dD, dO, dOP, dM, dE = _contrib(j, lam, v, w, kk)
            cD += tl.sum(dD, 0)
            cO += tl.sum(dO, 0)
            cOP += tl.sum(dOP, 0)
            cM += tl.sum(dM, 0)
            cE += tl.sum(dE, 0)
            cT += tl.sum(w, 0)
            dc = tl.load(GC + idx, mask=msk, other=0.0)
            gt += tl.sum(tl.where(oh, (dc * (1 - lam))[:, None], 0.0) + tl.where(oh1, (dc * lam)[:, None], 0.0), 0)
        z2 = tl.zeros([BLOCK, KP], tl.float32)
        z1 = tl.zeros([BLOCK], tl.float32)
        D = z2 + cD[None, :]
        O = z2 + cO[None, :]
        OP = z2 + cOP[None, :]
        M = z2 + cM[None, :]
        if NEFF:
            dl = z1 + tl.load(DEFF + pid)
        else:
            dl = z1
        NRHO, PIV, LL, RET, DELTA, DNEXT, DPREV, p, s, Fs, ok, Vp, nus, ps, Ss, Tc, sa, sb, m1, sig2, gam, sp = \
            _fit_fwd(D, O, OP, M, z1 + cE, z1 + cT, rho, rhon, rhop, eo, tau, kk, K, dl, NEFF)
        first = (ln == 0)[:, None]
        GT = tl.where(first, gt[None, :], 0.0)                          # the fit is shared: its gradient once
        GD, GO, GM, gE, GR, go = _fit_bwd(GT, D, O, OP, M, rho, rhon, NRHO, PIV, LL, RET, DELTA, DNEXT, DPREV, s,
                                          ok, Vp, nus, ps, Ss, Tc, sa, sb, m1, sig2, gam, sp, kk, K, dl, NEFF)
        gr = tl.sum(GR, 0)
        gom = tl.sum(go, 0)
        RD = z2 + tl.sum(GD, 0)[None, :]
        RO = z2 + tl.sum(GO, 0)[None, :]
        RM = z2 + tl.sum(GM, 0)[None, :]
        RE = z1 + tl.sum(gE, 0)
        TH = z2 + tl.sum(tl.where(first, rho[None, :] + s[:, None] * DELTA, 0.0), 0)[None, :]
        for ch in range(0, NCH):
            idx, msk, v, w, j, lam = _token(U, V, W, b, k, ch * BLOCK + ln, N, R, swb, swn, swk, lo, inv_h, K)
            oh = kk[None, :] == j[:, None]
            oh1 = kk[None, :] == (j + 1)[:, None]
            dc = tl.load(GC + idx, mask=msk, other=0.0)
            _token_grads(GC, DU, DV, idx, msk, v, w, lam, oh, oh1, TH, RD, RO, RM, RE, dc, inv_h)
    tl.store(GRHO + pid * KP + kk, gr)
    tl.store(GOM + pid, gom)


BLOCK = 32
NUM_WARPS = 8                       # (BLOCK, warps) = (32, 8) or (16, 4) keep the tiles in registers (sm_80)


class _Fused(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, v, w, rho, omega, lo, hi, tau, causal, deff=None):
        u, v = u.contiguous(), v.contiguous()
        B, N, R = u.shape
        K = rho.shape[-1]
        KP = triton.next_power_of_2(K)
        w = w.expand(B, N, R)
        rho = rho.expand(R, K).contiguous()
        eo = omega.detach().exp().reshape(1)
        NCH = triton.cdiv(N, BLOCK)
        h = np.float32((hi - lo) / (K - 1))
        inv_h = float(np.float32(1.0) / h)                             # torch divides by a scalar as * (1/h)
        c = torch.empty_like(u)
        st = torch.empty((B, N, R) if causal else (B, R), device=u.device, dtype=u.dtype)
        so, po, fo = st, torch.empty_like(st), torch.empty_like(st)
        car = torch.empty((B * R, NCH, 5, KP) if causal else (1,), device=u.device, dtype=u.dtype)
        neff = deff is not None
        if neff:                                                       # (B, r) pooled, (B, N, r) causal; detached
            deff = deff.detach().to(u.dtype).expand((B, N, R) if causal else (B, R)).contiguous()
            bo = torch.empty_like(st)
        else:
            deff, bo = car, car                                        # unused (NEFF=False compiles them out)
        _fwd_kernel[(B * R,)](u, v, w, rho, eo, c, so, po, fo, car, deff, bo, N, R, NCH, *w.stride(), float(lo),
                              inv_h, float(tau), K=K, KP=KP, BLOCK=BLOCK, CAUSAL=bool(causal), NEFF=neff,
                              num_warps=NUM_WARPS)
        ctx.save_for_backward(u, v, w, rho, eo, car, deff)
        ctx.cfg = (lo, inv_h, tau, causal, K, KP, NCH, neff)
        ctx.mark_non_differentiable(so, po, fo, bo)
        return c, so, po, fo, bo

    @staticmethod
    def backward(ctx, gc, *_):
        u, v, w, rho, eo, car, deff = ctx.saved_tensors
        lo, inv_h, tau, causal, K, KP, NCH, neff = ctx.cfg
        B, N, R = u.shape
        gc = gc.contiguous()
        du, dv = torch.empty_like(u), torch.empty_like(v)
        grho = torch.empty((B, R, KP), device=u.device, dtype=u.dtype)
        gom = torch.empty((B * R,), device=u.device, dtype=u.dtype)
        _bwd_kernel[(B * R,)](u, v, w, rho, eo, gc, car, deff, du, dv, grho, gom, N, R, NCH, *w.stride(), float(lo),
                              inv_h, float(tau), K=K, KP=KP, BLOCK=BLOCK, CAUSAL=bool(causal), NEFF=neff,
                              num_warps=NUM_WARPS)
        return du, dv, None, grho.sum(0)[:, :K], gom.sum(), None, None, None, None, None


def fused_codes(u, v, w, rho, omega, lo, hi, tau, causal, deff=None):
    """u, v, w (B, N, r) fp32 CUDA; rho (r, K) or broadcastable; omega (). Returns codes and (s, p, F).
    deff (LeNA-S): design effect, (B, r) pooled or (B, N, r) causal, detached; adds bend = max_k |s delta_k| to the dict."""
    c, s, p, Fs, bo = _Fused.apply(u, v, w, rho, omega, lo, hi, tau, causal, deff)
    d = dict(s=s, p=p, F=Fs)
    if deff is not None:
        d["bend"] = bo
    return c, d
