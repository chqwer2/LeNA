"""Unit tests for lena_aff.py (CPU, fp64)."""
import os
import sys

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
import lena_v7 as LV  # noqa: E402
from lena_aff import LeNAAff, LoRAINorm  # noqa: E402

torch.manual_seed(0); torch.set_default_dtype(torch.float64)
d, r, K = 24, 4, 16


def mk(cls, **kw):
    torch.manual_seed(1)
    base = nn.Linear(d, 20)
    L = cls(base, r=r, alpha=8, **kw)
    with torch.no_grad():
        L.B.normal_();
        if hasattr(L, "rho"):
            L.rho.add_(0.3 * torch.randn_like(L.rho).cumsum(-1))
    return L


ok = True
def check(name, cond):
    global ok
    print(("PASS " if cond else "FAIL ") + name); ok &= bool(cond)


for causal, N in ((False, 196), (False, 10), (True, 12)):
    L = mk(LeNAAff, K=K, causal=causal, use_checkpoint=False)
    x = torch.randn(3, N, d) * 2
    y = L(x)
    check(f"aff shape causal={causal} N={N}", y.shape == (3, N, 20))
    # s = 0 (fixed) equals LeNALinear fixed with the same weights
    R = LV.LeNALinear(L.base, r=r, alpha=8, K=K, causal=causal, use_checkpoint=False, mode="fixed")
    R.load_state_dict(L.state_dict())
    with LV.lena_override(mode="fixed"):
        ya = L(x)
    check(f"aff s=0 == fixed rho causal={causal}", torch.allclose(ya, R(x), atol=1e-10))
    # plain (s = 1): codes = LS affine of v on [rho(u), 1] over the tokens (non-causal) / prefix (causal, last pos)
    L.log_stats = True
    with LV.lena_override(mode="plain"):
        L(x)
    th = L.last["theta"]
    u, v = L.last["u"], L.last["v"]
    j, lam = LV.hat_index(u, L.lo, L.hi, K)
    ru = LV.spline_eval(L.rho.expand(3, 1, r, K) if not causal else L.rho.expand(3, N, r, K), j, lam)
    if causal:
        ru_, v_, th_ = ru, v, th[:, -1]
    else:
        ru_, v_, th_ = ru, v, th
    X = torch.stack([ru_, torch.ones_like(ru_)], -1).permute(0, 2, 1, 3)     # (B, r, N, 2)
    coef = torch.linalg.lstsq(X, v_.permute(0, 2, 1).unsqueeze(-1)).solution.squeeze(-1)
    want = coef[..., :1] * L.rho + coef[..., 1:]
    check(f"aff plain == token LS causal={causal} (err {(th_ - want).abs().max():.1e})", torch.allclose(th_, want, atol=1e-3))
    L.log_stats = False
    # gradients
    L.zero_grad(); L(x).square().mean().backward()
    check(f"aff grads causal={causal}", all(getattr(L, n).grad is not None and getattr(L, n).grad.abs().sum() > 0
                                            for n in ("A1", "A2", "B", "rho")))
    if causal:   # outputs at t do not depend on later tokens
        x2 = x.clone(); x2[:, 7] += 5
        check("aff causal: outputs < 7 unchanged", torch.allclose(L(x)[:, :7], L(x2)[:, :7]))

for causal, N in ((False, 196), (True, 12)):
    L = mk(LoRAINorm)
    with torch.no_grad():
        L.gamma.normal_(); L.beta.normal_()
    x = torch.randn(2, N, d) + 3
    with LV.lena_context(causal=causal):
        y = L(x)
        u = x @ L.A1.t()
        if causal:
            n = torch.arange(1, N + 1).view(1, N, 1)
            mu = u.cumsum(1) / n; var = (u * u).cumsum(1) / n - mu ** 2
        else:
            mu = u.mean(1, keepdim=True); var = u.var(1, unbiased=False, keepdim=True)
        want = L.base(x) + L.scaling * ((u - mu) / (var + L.eps).sqrt() * L.gamma + L.beta) @ L.B.t()
        check(f"inorm matches manual causal={causal}", torch.allclose(y, want, atol=1e-8))
        L.zero_grad(); L(x).square().mean().backward()
        check(f"inorm grads causal={causal}", all(getattr(L, n).grad.abs().sum() > 0 for n in ("A1", "B", "gamma", "beta")))
    # inorm is in adapter_tensors (saved) and LeNAAff in lena_layers
check("saved tensors", any(k.endswith("gamma") for k in LV.adapter_tensors(nn.Sequential(mk(LoRAINorm)))))
check("lena_layers sees LeNAAff", len(LV.lena_layers(nn.Sequential(mk(LeNAAff, K=K)))) == 1)
print("ALL PASSED" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
