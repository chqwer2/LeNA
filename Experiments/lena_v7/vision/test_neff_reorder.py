"""Beam reorder of the causal n_eff stream state: prompt pass, reorder(perm), then single-token steps == full pass on the reordered sequences."""
import sys
sys.path.insert(0, "/mnt/scratch2/users/hchen/FLoRA/Experiments/lena_v7"); sys.path.insert(0, "/mnt/scratch2/users/hchen/FLoRA/Experiments/lena_v7/vision")
import torch, torch.nn as nn
import lena_v7 as LV, lena_mri_ext as X
torch.manual_seed(0)
m = nn.Sequential(nn.Linear(16, 16))
LV.inject(m, ["0"], adapter="lena", r=4, alpha=4, K=16, causal=True, mode="lena", use_checkpoint=False)
X.install(m, neff=True)
L = LV.lena_layers(m)[0]; L.training = False
B, P, N = 4, 20, 10
x1 = torch.randn(B, P, 4).cumsum(1) * 0.3; x2 = torch.randn(B, N, 4) * 0.3 + x1[:, -1:]
perm = torch.tensor([2, 0, 3, 1])
LV._CTX.stream = True; L._state = None
X.prefix_lag1(x1, torch.ones_like(x1), L)
L._state = (torch.zeros(B, 1),)                     # mimic a live layer state so reorder_state runs
L.reorder_state(perm)
outs = []
for t in range(N):
    r, _ = X.prefix_lag1(x2[:, t:t + 1], torch.ones(B, 1, 4), L); outs.append(r)
LV._CTX.stream = False
got = torch.cat(outs, 1)
full = torch.cat([x1[perm], x2], 1)
ref, _ = X.prefix_lag1(full, torch.ones_like(full))
d = (got - ref[:, P:]).abs().max().item()
print("max diff", d); assert d < 1e-5
print("REORDER TEST PASSED")
