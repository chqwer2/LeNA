"""Sanity check of non-causal record/inject: injecting each input's own recorded statistics must reproduce the normal
forward pass exactly; also report injecting the batch-pooled sums and another input's statistics."""
import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from lena_v7 import lena_layers, lena_override  # noqa
from camelyon import ParquetImages, MEAN, STD  # noqa
from diag_camelyon import build, D  # noqa
from torchvision import transforms as T  # noqa
torch.manual_seed(0)
tf = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BILINEAR), T.ToTensor(), T.Normalize(MEAN, STD)])
ds = ParquetImages([os.path.join(D, "validation.pkl")], tf); idx = [i for i, c in enumerate(ds.cs) if c == 0][:64]
x = torch.stack([ds[i][0] for i in idx]).cuda(); y = torch.tensor([ds[i][1] for i in idx])
m = build("lena", "runs/cam_lena_lr5e-5/best.pt", "cuda"); L = lena_layers(m)
with torch.no_grad():
    base = m(x).float()
    with lena_override(stats="record"): rec = m(x).float()
    st = [tuple(t.clone() for t in l._rec) for l in L]
    for l, s in zip(L, st): l._inj = s
    with lena_override(stats="inject"): own = m(x).float()
    for l, s in zip(L, st): l._inj = tuple(t.roll(1, 0) for t in s)
    with lena_override(stats="inject"): other = m(x).float()
    for l, s in zip(L, st): l._inj = tuple(t.sum(0, keepdim=True) for t in s)
    with lena_override(stats="inject"): pooled = m(x).float()
acc = lambda o: (o.argmax(-1).cpu() == y).float().mean().item()
print("max|record-base|", (rec - base).abs().max().item(), "max|inject_own-base|", (own - base).abs().max().item())
print("acc base", acc(base), "inject own", acc(own), "inject other patch", acc(other), "inject batch-pooled", acc(pooled))
print("shapes", [tuple(t.shape) for t in st[0]])
