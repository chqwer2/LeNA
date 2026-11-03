"""Unit tests for lena_qt.LeNAQT (CPU, float64)."""
import copy

import torch
import torch.nn as nn

import lena_v7 as LV
from lena_qt import LeNAQT, weighted_quantiles, prefix_quantiles, interp_clamped

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)


def layer(causal, d=24, r=4):
    L = LeNAQT(nn.Linear(d, d), r=r, alpha=8, K=16, mode="lena", causal=causal, use_checkpoint=False)
    nn.init.normal_(L.B, std=0.1)
    return L


def test_quantiles():
    u = torch.randn(3, 500, 4); w = torch.ones_like(u)
    q = weighted_quantiles(u, w)
    ref = torch.quantile(u, (torch.arange(1, 17) - 0.5) / 16, dim=1).permute(1, 2, 0)
    assert (q - ref).abs().max() < 0.15, (q - ref).abs().max()   # midpoint-CDF vs torch (N-1)tau convention differ by ~1 order-statistic gap
    assert (q[..., 1:] >= q[..., :-1]).all()
    print("ok weighted quantiles ~ torch.quantile (max diff %.3f), sorted" % float((q - ref).abs().max()))


def test_monotone_T():
    L = layer(False); L.train()
    x = torch.randn(2, 196, 24)
    L(x)                                                              # sets q_ref
    L.eval()
    u = torch.randn(2, 196, 4) * 1.5 + 0.3
    qx = weighted_quantiles(u, torch.ones_like(u))
    grid = torch.linspace(-6, 6, 400).expand(2, 4, 400)
    T = interp_clamped(grid, qx, L.q_ref.expand_as(qx))
    assert (T[..., 1:] >= T[..., :-1] - 1e-12).all()
    print("ok T_x is monotone (non-decreasing)")


def test_s0_fixed():
    for causal in (False, True):
        L = layer(causal)
        ref = LV.LeNALinear(copy.deepcopy(L.base), r=4, alpha=8, K=16, mode="fixed", causal=causal, use_checkpoint=False)
        with torch.no_grad():
            ref.A1.copy_(L.A1); ref.B.copy_(L.B); ref.rho.copy_(L.rho)
        x = torch.randn(2, 49 if not causal else 9, 24)
        with LV.lena_override(mode="fixed"):
            y = L(x)
        assert torch.allclose(y, ref(x), atol=1e-10)
    print("ok s = 0 gives the plain fixed-curve adapter")


def test_causal():
    B, N, r = 2, 20, 4
    u = torch.randn(B, N, r); w = torch.ones_like(u)
    q0, c0 = prefix_quantiles(u, w, -3.0, 3.0, 16)
    for t in (0, 7, 19):
        u2 = u.clone(); u2[:, t] += 2.0
        q1, c1 = prefix_quantiles(u2, w, -3.0, 3.0, 16)
        assert torch.allclose(q0[:, : t + 1], q1[:, : t + 1]) and torch.allclose(c0, c1), f"stats at <= {t} saw token {t}"
        if t < N - 1:
            assert not torch.allclose(q0[:, t + 1:], q1[:, t + 1:])
    L = layer(True); L.train(); L(torch.randn(2, N, 24)); L.eval()
    x = torch.randn(2, N, 24); y0 = L(x)
    for t in (3, 12):
        x2 = x.clone(); x2[:, t] += 1.0
        y1 = L(x2)
        assert torch.allclose(y0[:, :t], y1[:, :t]), "earlier outputs changed"
    print("ok causal: statistics at t use only tokens < t; outputs before t unchanged")


def test_grad_and_ref():
    L = layer(False); L.train()
    x = torch.randn(2, 196, 24, requires_grad=True)
    L(x).sum().backward()
    for n in ("A1", "B", "rho", "kappa_raw"):
        g = getattr(L, n).grad
        assert g is not None and torch.isfinite(g).all(), n
    assert L.q_ref_init.item() == 1 and torch.isfinite(L.q_ref).all()
    print("ok gradients reach A1, B, rho, kappa; q_ref initialised")


if __name__ == "__main__":
    test_quantiles(); test_monotone_T(); test_s0_fixed(); test_causal(); test_grad_and_ref()
    print("ALL TESTS PASSED")
