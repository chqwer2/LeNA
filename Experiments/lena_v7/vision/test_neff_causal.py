"""CPU test of the causal R1 patch: causality, stream == full, padding, deff in the fit, s vs plain."""
import os, sys
sys.path.insert(0, "/mnt/scratch2/users/hchen/FLoRA/Experiments/lena_v7"); sys.path.insert(0, "/mnt/scratch2/users/hchen/FLoRA/Experiments/lena_v7/vision")
import torch, torch.nn as nn
import lena_v7 as LV, lena_mri_ext as X
torch.manual_seed(0)
B, T, r = 3, 40, 4
u = torch.randn(B, T, r).cumsum(1) * 0.3          # autocorrelated sequence
w = torch.ones(B, T, r); w[0, :5] = 0              # left padding on input 0
rho, n = X.prefix_lag1(u, w)
# causality
u2 = u.clone(); u2[:, 20] += 5
rho2, _ = X.prefix_lag1(u2, w)
assert torch.allclose(rho[:, :20], rho2[:, :20]), "future token changed the past"
assert not torch.allclose(rho[:, 20:], rho2[:, 20:])
# stream == full
class Dummy: pass
L = Dummy(); L.training = False; L._state = None
LV._CTX.stream = True
outs = []
chunks = [(0, 25)] + [(t, t + 1) for t in range(25, T)]
for a_, b_ in chunks:
    rr, nn_ = X.prefix_lag1(u[:, a_:b_], w[:, a_:b_], L)
    outs.append(rr); L._state = "set"
LV._CTX.stream = False
rs = torch.cat(outs, 1)
print("stream max diff", (rs - rho).abs().max().item())
assert torch.allclose(rs, rho, atol=1e-5)
print("padding: n of input0 at t=4", n[0, 4, 0].item(), "rho", rho[0, 4, 0].item(), "| rho mean t>=16", rho[:, 16:].mean().item())
# full layer: plain vs neff s on a causal LeNA layer
lin = nn.Linear(32, 32)
torch.manual_seed(1)
x = torch.randn(2, 64, 32).cumsum(1) * 0.2
def run(neff):
    X.EXT["neff"] = neff; X.EXT["deff"] = None; X.EXT["acc"] = {}
    m = nn.Sequential(nn.Linear(32, 32))
    LV.inject(m, ["0"], adapter="lena", r=4, alpha=4, K=16, causal=True, mode="lena", use_checkpoint=False)
    torch.manual_seed(2)
    for L_ in LV.lena_layers(m):
        for p_ in L_.parameters(): p_.data.normal_(0, 0.3) if p_.dim() > 1 else None
    X.install(m, neff=neff); X.EXT["acc"] = {}
    with torch.no_grad(): y = m(x)
    a = X.EXT["acc"]; return y, a["s"] / a["n"]
y0, s0 = run(False)
LV.prefix_stats = X._ORIG["prefix_stats"]
y1, s1 = run(True)
print("plain s %.3f  neff s %.3f  out diff %.3g" % (s0, s1, (y0 - y1).abs().max().item()))
assert s1 < s0
print("ALL PASSED")
