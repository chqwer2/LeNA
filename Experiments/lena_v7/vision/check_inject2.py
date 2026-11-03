"""Isolate why diag_pooled collapses: record pooled stats on batch A (separate pass), inject into batch B, with and
without bf16 autocast, for pools of 1, 64, 128 patches of the same hospital."""
import contextlib, os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from lena_v7 import lena_layers, lena_override  # noqa
from camelyon import ParquetImages, MEAN, STD  # noqa
from diag_camelyon import build, D  # noqa
from torchvision import transforms as T  # noqa
tf = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BILINEAR), T.ToTensor(), T.Normalize(MEAN, STD)])
ds = ParquetImages([os.path.join(D, "validation.pkl")], tf); pos = [i for i, c in enumerate(ds.cs) if c == 0 and ds.ys[i] == 1]; neg = [i for i, c in enumerate(ds.cs) if c == 0 and ds.ys[i] == 0]
B = pos[:32] + neg[:32]; A = pos[32:96] + neg[32:96]
xb = torch.stack([ds[i][0] for i in B]).cuda(); yb = torch.tensor([ds[i][1] for i in B])
m = build("lena", "runs/cam_lena_lr5e-5/best.pt", "cuda"); L = lena_layers(m)
acc = lambda o: f"{(o.argmax(-1).cpu() == yb).float().mean().item():.3f} (pred=1: {(o.argmax(-1) == 1).float().mean().item():.2f})"
for amp in (False, True):
    ac = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if amp else contextlib.nullcontext
    with torch.no_grad(), ac():
        base = acc(m(xb).float()); out = [f"amp={amp} base {base}"]
        for n in (1, 64, 128):
            xa = torch.stack([ds[i][0] for i in A[:n]]).cuda()
            with lena_override(stats="record"): m(xa)
            st = [tuple(t.float().sum(0, keepdim=True) for t in l._rec) for l in L]
            for l, s in zip(L, st): l._inj = s
            with lena_override(stats="inject"): out.append(f"pool{n} {acc(m(xb).float())}")
            pass
        print(" | ".join(out), flush=True)
