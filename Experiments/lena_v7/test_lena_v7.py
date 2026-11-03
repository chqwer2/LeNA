"""Checks of lena_v7 against the paper's definitions. Run: python -m pytest -q test_lena_v7.py"""
import math
import torch
import torch.nn as nn
from lena_v7 import (hat_index, spline_eval, stats_dense, retained_solve, tri_mv, fit_and_step, LeNALinear,
                     lena_context, lena_override, inject_lena)

torch.manual_seed(0)
K, LO, HI = 16, -3.0, 3.0


def dense_n(diag, off):
    n = torch.diag_embed(diag)
    return n + torch.diag_embed(off, 1) + torch.diag_embed(off, -1)


def ref_retain(n, tau):
    """Literal App. A.1 recurrence."""
    Kk = n.shape[0]; J = []; dprev = None; prev = False
    for j in range(Kk):
        d = n[j, j] - (n[j, j - 1] ** 2 / dprev if (j > 0 and prev) else 0)
        prev = bool(d > tau)
        if prev:
            J.append(j); dprev = d
    return J


def basis_matrix(u):
    j, lam = hat_index(u, LO, HI, K)
    Phi = torch.zeros(u.shape[0], K, dtype=u.dtype)
    Phi[torch.arange(len(u)), j] = 1 - lam
    Phi[torch.arange(len(u)), j + 1] += lam
    return Phi


def test_identity_prior_reproduces_coordinate_including_extrapolation():
    u = torch.linspace(-6, 6, 101)
    j, lam = hat_index(u, LO, HI, K)
    knots = torch.linspace(LO, HI, K)
    assert torch.allclose(spline_eval(knots.expand(101, K), j, lam), u, atol=1e-5)


def test_stats_match_basis_matrix():
    u = torch.randn(200, dtype=torch.float64) * 1.5; v = torch.randn(200, dtype=torch.float64)
    j, lam = hat_index(u, LO, HI, K)
    diag, off, m, e = [t.sum(0) for t in stats_dense(j, lam, v, torch.ones_like(v), K)]
    Phi = basis_matrix(u)
    assert torch.allclose(dense_n(diag, off), Phi.T @ Phi)
    assert torch.allclose(m, Phi.T @ v) and torch.allclose(e, (v * v).sum())


def test_retention_and_solve_match_dense_reference():
    for trial in range(50):
        N = int(torch.randint(5, 120, ()).item())
        u = torch.randn(N, dtype=torch.float64) * (0.5 + 2 * torch.rand(()).item()) + torch.randn(()).item()
        v = torch.randn(N, dtype=torch.float64)
        j, lam = hat_index(u, LO, HI, K)
        diag, off, m, e = [t.sum(0) for t in stats_dense(j, lam, v, torch.ones_like(v), K)]
        rho = torch.randn(K, dtype=torch.float64)
        rhs = m - tri_mv(diag, off, rho)
        delta, ret = retained_solve(diag, off, rhs, 1.0)
        n = dense_n(diag, off)
        J = ref_retain(n, 1.0)
        assert ret.nonzero().flatten().tolist() == J
        if J:
            nJ = n[J][:, J]
            assert torch.linalg.eigvalsh(nJ).min() > 0
            ref = torch.zeros(K, dtype=torch.float64); ref[J] = torch.linalg.solve(nJ, rhs[J])
            assert torch.allclose(delta, ref, atol=1e-8)
        else:
            assert delta.abs().max() == 0


def test_rss_identity_and_step_formula():
    N = 196
    u = torch.randn(N, dtype=torch.float64); v = u + 0.8 * torch.relu(u - 0.2) + torch.randn(N, dtype=torch.float64)
    j, lam = hat_index(u, LO, HI, K)
    diag, off, m, e = [t.sum(0) for t in stats_dense(j, lam, v, torch.ones_like(v), K)]
    rho = torch.linspace(LO, HI, K, dtype=torch.float64)
    theta, d = fit_and_step(diag, off, m, e, torch.tensor(float(N), dtype=torch.float64), rho,
                            torch.tensor(0.0, dtype=torch.float64), 1.0, "lena")
    # explicit residuals
    Phi = basis_matrix(u)
    J = ref_retain(dense_n(diag, off), 1.0); p = len(J); nu = N - p
    r = v - Phi @ rho
    dJ = torch.linalg.lstsq(Phi[:, J], r).solution
    RSS = ((r - Phi[:, J] @ dJ) ** 2).sum(); S = ((Phi[:, J] @ dJ) ** 2).sum()
    gam = (p - 2) / p * nu / (nu + 2)
    Fs = S / (p * RSS / nu)
    s_ref = max(1 - gam / Fs.item(), 0.0)
    assert abs(d["s"].item() - s_ref) < 1e-9 and abs(d["F"].item() - Fs.item()) < 1e-6
    full = rho.clone(); full[J] += s_ref * dJ
    assert torch.allclose(theta, full, atol=1e-8)


def test_guard_short_inputs_keep_prior():
    u = torch.randn(1, 5, 4); v = torch.randn(1, 5, 4)
    j, lam = hat_index(u, LO, HI, K)
    diag, off, m, e = [t.sum(1) for t in stats_dense(j, lam, v, torch.ones_like(v), K)]
    rho = torch.randn(4, K)
    theta, d = fit_and_step(diag, off, m, e, torch.full((1, 4), 5.0), rho, torch.tensor(0.0), 1.0, "lena")
    assert torch.equal(theta, rho.expand_as(theta)) and (d["s"] == 0).all()


def make_layer(mode="lena", causal=False, d_in=24, d_out=20, r=4):
    base = nn.Linear(d_in, d_out)
    L = LeNALinear(base, r=r, alpha=2 * r, mode=mode, causal=causal, use_checkpoint=False).double()
    with torch.no_grad():
        L.B.normal_(); L.rho.add_(0.3 * torch.randn_like(L.rho)); L.A1.mul_(4); L.A2.mul_(4)
    return L


def test_zero_B_gives_frozen_output():
    base = nn.Linear(24, 20).double()
    L = LeNALinear(base, r=4).double()
    x = torch.randn(2, 30, 24, dtype=torch.float64)
    assert torch.allclose(L(x), base(x))


def test_fixed_identity_prior_equals_lora():
    L = make_layer(mode="fixed")
    with torch.no_grad():
        L.rho.copy_(torch.linspace(LO, HI, K).repeat(4, 1))
    x = torch.randn(2, 30, 24, dtype=torch.float64)
    lora = L.base(x) + L.scaling * (x @ L.A1.T) @ L.B.T
    assert torch.allclose(L(x), lora, atol=1e-6)


def test_causal_equals_bidirectional_on_prefixes():
    Lc = make_layer(causal=True); Lb = make_layer(causal=False); Lb.load_state_dict(Lc.state_dict())
    x = torch.randn(1, 40, 24, dtype=torch.float64)
    yc = Lc(x)
    for t in (3, 10, 25, 40):
        yb = Lb(x[:, :t])
        assert torch.allclose(yc[:, t - 1], yb[:, t - 1], atol=1e-6), t


def test_padding_is_ignored():
    for causal in (False, True):
        L = make_layer(causal=causal)
        x = torch.randn(1, 30, 24, dtype=torch.float64)
        xp = torch.cat([x, torch.randn(1, 7, 24, dtype=torch.float64)], 1)
        mask = torch.cat([torch.ones(1, 30), torch.zeros(1, 7)], 1)
        with lena_context(attention_mask=mask):
            yp = L(xp)
        assert torch.allclose(yp[:, :30], L(x), atol=1e-6)


def test_gradients_flow_and_gradcheck():
    L = make_layer(r=2, d_in=6, d_out=5)
    x = torch.randn(1, 50, 6, dtype=torch.float64)
    L(x).pow(2).sum().backward()
    for n_, p in L.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all(), n_
    # gradcheck through the fit on the response read
    def g(A2):
        u = x @ L.A1.T; v = x @ A2.T
        c, _ = L._codes(u, v, torch.ones_like(u))
        return c
    A2 = L.A2.detach().clone().requires_grad_(True)
    assert torch.autograd.gradcheck(g, (A2,), eps=1e-6, atol=1e-5)


def test_shrinkage_beats_plain_fit_in_domain():
    """Small version of the controlled check: noisy short inputs, prior = true mean curve."""
    M, N = 400, 64
    a = 0.5 + torch.rand(M, 1); t = torch.rand(M, 1) - 0.5
    u = torch.randn(M, N, dtype=torch.float64)
    y = u + a * torch.relu(u - t)
    v = y + torch.randn(M, N, dtype=torch.float64)
    j, lam = hat_index(u, LO, HI, K)
    diag, off, m, e = [q.sum(1) for q in stats_dense(j, lam, v, torch.ones_like(v), K)]
    knots = torch.linspace(LO, HI, K, dtype=torch.float64)
    rho = (knots + 1.0 * torch.relu(knots)).expand(M, K)
    err = {}
    for mode in ("fixed", "plain", "lena"):
        th, _ = fit_and_step(diag, off, m, e, torch.full((M,), float(N), dtype=torch.float64), rho,
                             torch.tensor(0.0, dtype=torch.float64), 1.0, mode)
        err[mode] = ((spline_eval(th.unsqueeze(1).expand(M, N, K), j, lam) - y) ** 2).mean().item()
    assert err["lena"] < err["plain"], err


def test_inject():
    net = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 3))
    names = inject_lena(net, ["0"], r=2)
    assert names == ["0"] and isinstance(net[0], LeNALinear)
    net(torch.randn(2, 5, 8)).sum().backward()


def test_variants_run_and_start_at_frozen_output():
    from lena_v7 import adapter_state_dict, load_adapter_state_dict
    x = torch.randn(2, 30, 24, dtype=torch.float64)
    for kw in [dict(mode="predictor"), dict(prior="u_pred"), dict(prior="identity"), dict(freeze_A2=True),
               dict(learn_omega=False), dict(dora=True), dict(mode="plain", K=2), dict(causal=True, prior="u_pred"),
               dict(causal=True, mode="predictor")]:
        base = nn.Linear(24, 20).double()
        L = LeNALinear(base, r=4, use_checkpoint=False, **kw).double()
        assert torch.allclose(L(x), base(x)), kw                      # B = 0 at init
        with torch.no_grad():
            L.B.normal_()
        L.train(); L(x).sum().backward()
        assert L.A1.grad is not None, kw
        if kw.get("freeze_A2"):
            assert not L.A2.requires_grad
        net = nn.Sequential(L)
        sd = adapter_state_dict(net)
        assert not any(".base." in k for k in sd) and "0.B" in sd
        load_adapter_state_dict(net, sd)


def test_checkpoint_path_matches():
    L = make_layer(); x = torch.randn(2, 30, 24, dtype=torch.float64)
    L.eval(); y0 = L(x)
    L.use_checkpoint = True; L.train()
    y1 = L(x)
    assert torch.allclose(y0, y1)
    y1.sum().backward()


def test_overrides_match_constructed_variants():
    L = make_layer(causal=True)
    x = torch.randn(2, 30, 24, dtype=torch.float64)
    for mode in ("fixed", "plain"):
        Lm = make_layer(mode=mode, causal=True); Lm.load_state_dict(L.state_dict())
        with lena_override(mode=mode):
            assert torch.allclose(L(x), Lm(x))
    Li = make_layer(causal=True); Li.load_state_dict(L.state_dict())
    with torch.no_grad():
        Li.rho.copy_(torch.linspace(LO, HI, K).repeat(4, 1))
    with lena_override(prior="identity"):
        assert torch.allclose(L(x), Li(x))


def test_record_inject_own_statistics_reproduces_last_position():
    L = make_layer(causal=True)
    x = torch.randn(1, 30, 24, dtype=torch.float64)
    with lena_override(stats="record"):
        y = L(x)
    L._inj = L._rec
    with lena_override(stats="inject"):
        yi = L(x)
    assert torch.allclose(yi[:, -1], y[:, -1])
    assert not torch.allclose(yi[:, 5], y[:, 5])              # earlier positions now see the whole input
    xo = torch.randn(1, 30, 24, dtype=torch.float64)          # another input's statistics change the output
    with lena_override(stats="inject"):
        assert not torch.allclose(L(xo)[:, -1], L.base(xo)[:, -1] + 0)


# ---------------------------------------------------------------------------------------------------------------
# Fused Triton path (lena_v7.set_triton): skipped without CUDA / Triton
# ---------------------------------------------------------------------------------------------------------------
def _triton_ok():
    import importlib.util
    if torch.cuda.is_available() and importlib.util.find_spec("triton") is not None:
        return True
    print("  skip: no CUDA / Triton")
    return False


def _fused_vs_ref(L, u, v, w, g):
    import lena_v7
    out = {}
    for on in (False, True):
        lena_v7.set_triton(on)
        try:
            uu, vv = u.clone().requires_grad_(True), v.clone().requires_grad_(True)
            L.zero_grad()
            c, d = L._codes(uu, vv, w)
            (c * g).sum().backward()
            out[on] = (c.detach(), d["s"].detach(), uu.grad, vv.grad, L.rho.grad.clone(), L.omega.grad.clone())
        finally:
            lena_v7.set_triton(False)
    return out[False], out[True]


def _rel(a, b):
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def test_triton_matches_reference():
    if not _triton_ok():
        return
    gen = torch.Generator().manual_seed(1)
    cases = [(3, 70, 8, 1.0, False), (2, 200, 6, 2.5, True), (2, 6, 4, 1.0, False), (4, 33, 5, 4.0, True),
             (1, 1, 3, 1.0, False)]                                  # (B, N, r, u scale, padding)
    for causal in (True, False):
        for B, N, r, scale, pad in cases:
            L = LeNALinear(nn.Linear(8, 8), r=r, causal=causal, use_checkpoint=False).cuda()
            with torch.no_grad():
                L.rho.add_(0.3 * torch.randn(L.rho.shape, generator=gen).cuda()); L.omega.fill_(0.3)
            u = (torch.randn(B, N, r, generator=gen) * scale).cuda()
            v = (u + 0.8 * torch.relu(u - 0.2) + 0.5 * torch.randn(B, N, r, generator=gen).cuda())
            w = torch.ones(B, N, r, device="cuda")
            if pad:
                for b in range(B):
                    w[b, N - (b * 7) % max(N // 2, 1):] = 0
            g = torch.randn(B, N, r, generator=gen).cuda()
            ref, fus = _fused_vs_ref(L, u, v, w, g)
            tag = (causal, B, N, r, scale, pad)
            assert (fus[0] - ref[0]).abs().max() < 1e-4 * max(1.0, ref[0].abs().max().item()), tag
            assert (fus[1] - ref[1]).abs().max() < 1e-4, tag                   # step s
            for name, a, b in zip(("u", "v", "rho", "omega"), fus[2:], ref[2:]):
                if b.norm() > 0:
                    assert _rel(a, b) < 1e-3, (tag, name, _rel(a, b))
                else:
                    assert a.abs().max() < 1e-6, (tag, name)


def test_triton_layer_training_step_matches():
    if not _triton_ok():
        return
    import lena_v7
    torch.manual_seed(3)
    for causal in (True, False):
        L = LeNALinear(nn.Linear(64, 48).cuda(), r=8, alpha=16, causal=causal).cuda()   # inner checkpoint on
        with torch.no_grad():
            L.B.normal_(0, 0.1); L.A1.mul_(3); L.A2.mul_(3)
        x = torch.randn(2, 90, 64, device="cuda")
        mask = torch.ones(2, 90, device="cuda"); mask[1, 60:] = 0
        grads = []
        for on in (False, True):
            lena_v7.set_triton(on)
            try:
                L.zero_grad()
                with lena_context(attention_mask=mask):
                    y = L(x)
                y.pow(2).sum().backward()
                grads.append([y.detach()] + [p.grad.clone() for p in (L.A1, L.A2, L.B, L.rho, L.omega)])
            finally:
                lena_v7.set_triton(False)
        assert torch.allclose(grads[1][0], grads[0][0], atol=1e-4, rtol=1e-4), causal
        for a, b in zip(grads[1][1:], grads[0][1:]):
            assert _rel(a, b) < 1e-3, (causal, _rel(a, b))
