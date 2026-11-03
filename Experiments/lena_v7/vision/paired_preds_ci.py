"""Paired image-bootstrap CI of accuracy differences from preds.pt files (same images, same order).
  python paired_preds_ci.py <key> "<name>|<a_dir>|<b_dir>" ...   key = target | id_val | learned/<set> ...  (delta = a − b, %)"""
import sys, torch, numpy as np
rng = np.random.default_rng(0)
key = sys.argv[1]
for spec in sys.argv[2:]:
    name, a, b = spec.split("|")
    A, B = torch.load(a + "/preds.pt"), torch.load(b + "/preds.pt")
    keys = [key] if key != "clip" else [k for k in A if k.startswith("learned/")]
    for k in keys:
        x, y = A[k].numpy().astype(float), B[k].numpy().astype(float)
        assert len(x) == len(y)
        d = x - y; n = len(d)
        bs = [d[rng.integers(0, n, n)].mean() for _ in range(2000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"{name:28s} {k:16s} a {100*x.mean():6.2f} b {100*y.mean():6.2f} Δ {100*d.mean():+6.2f} [{100*lo:+6.2f}, {100*hi:+6.2f}] n {n}" + (" *" if lo > 0 or hi < 0 else ""))
