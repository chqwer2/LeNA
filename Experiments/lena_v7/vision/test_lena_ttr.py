"""Unit tests for lena_ttr.LeNATTR (CPU, float64)."""
import copy

import torch
import torch.nn as nn

import lena_v7 as LV
from lena_ttr import LeNATTR, halves

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)


def layer(causal, d=24, r=4):
    L = LeNATTR(nn.Linear(d, d), r=r, alpha=8, K=16, mode="lena", causal=causal, use_checkpoint=False)
    nn.init.normal_(L.B, std=0.1)
    return L


def test_shapes():
    for causal, N in ((False, 196), (False, 10), (True, 12)):
        L = layer(causal)
        y = L(torch.randn(3, N, 24))
        assert y.shape == (3, N, 24), y.shape
        assert torch.isfinite(y).all()
    print("ok shapes")


def test_noncausal_crossfit():
    L = layer(False)
    B, N, r = 2, 196, 4
    u, v, q, w = torch.randn(B, N, r), torch.randn(B, N, r), torch.randn(B, N, r), torch.ones(B, N, r)
    a = halves(N, u.device)
    i = int(torch.nonzero(a)[5])                                     # a token in half A
    v2 = v.clone(); v2[:, i] += 5.0; u2 = u.clone(); u2[:, i] += 1.0
    with LV.lena_override(mode="plain"):                             # the curve itself (s fixed at 1)
        c0, _ = L.codes_ttr(u, v, q, w, causal=False)
        c1, _ = L.codes_ttr(u2, v2, q, w, causal=False)
    same_half = a.clone()
    assert torch.allclose(c0[:, same_half], c1[:, same_half]), "a token's own half must not use its statistics"
    assert not torch.allclose(c0[:, ~same_half], c1[:, ~same_half]), "the other half should depend on it"
    print("ok non-causal: token", i, "does not reach its own half's curve (s fixed)")
    c0, d0 = L.codes_ttr(u, v, q, w, causal=False)
    c1, d1 = L.codes_ttr(u2, v2, q, w, causal=False)
    print("   note: with s from R2_cv the own half moves by", float((c0[:, a] - c1[:, a]).abs().max()),
          "- the held-out R2 that sets s for the B-fit is measured on half A (one detached scalar per input/channel)")


def test_causal_exclusive_prefix():
    L = layer(True)
    B, N, r = 2, 12, 4
    u, v, q, w = torch.randn(B, N, r), torch.randn(B, N, r), torch.randn(B, N, r), torch.ones(B, N, r)
    c0, _ = L.codes_ttr(u, v, q, w, causal=True)
    for t in (0, 5, 11):
        u2, v2 = u.clone(), v.clone(); u2[:, t] += 1.0; v2[:, t] += 5.0
        c1, _ = L.codes_ttr(u2, v2, q, w, causal=True)
        assert torch.allclose(c0[:, : t + 1], c1[:, : t + 1]), f"token {t} reaches itself or an earlier token"
        if t < N - 1:
            assert not torch.allclose(c0[:, t + 1:], c1[:, t + 1:])
    print("ok causal: no token sees itself or later tokens")


def test_s0_is_fixed_path():
    for causal in (False, True):
        L = layer(causal)
        ref = LV.LeNALinear(copy.deepcopy(L.base), r=4, alpha=8, K=16, mode="fixed", causal=causal, use_checkpoint=False)
        with torch.no_grad():
            ref.A1.copy_(L.A3); ref.A2.copy_(L.A2); ref.B.copy_(L.B); ref.rho.copy_(L.rho)
        x = torch.randn(2, 196 if not causal else 9, 24)
        with LV.lena_override(mode="fixed"):
            y = L(x)
        y_ref = ref(x)
        assert torch.allclose(y, y_ref, atol=1e-10), (y - y_ref).abs().max()
    print("ok s = 0 reproduces the fixed-rho path (curve on q)")


def test_grad():
    L = layer(False)
    x = torch.randn(2, 196, 24, requires_grad=True)
    L(x).sum().backward()
    for n in ("A1", "A2", "A3", "B", "rho"):
        g = getattr(L, n).grad
        assert g is not None and torch.isfinite(g).all(), n
    print("ok gradients reach A1, A2, A3, B, rho")


if __name__ == "__main__":
    test_shapes(); test_noncausal_crossfit(); test_causal_exclusive_prefix(); test_s0_is_fixed_path(); test_grad()
    print("ALL TESTS PASSED")
