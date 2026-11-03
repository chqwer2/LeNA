"""H15 CLIP: paired image bootstrap of a run's per-image correctness (preds.pt, key 'learned/<set>') vs references.
OOD mean = macro over v2, sketch, a, r; CI = 2000 resamples stratified by set (seed 0).
  python h15_clip_boot.py <out.csv> <name=run_dir> <ref1=dir> [<ref2=dir> ...]
"""
import csv
import sys

import numpy as np
import torch

OOD = ("v2", "sketch", "a", "r")


def load(d):
    p = torch.load(f"{d}/preds.pt", map_location="cpu")
    return {k.split("/", 1)[1]: v.float().numpy().reshape(-1) for k, v in p.items() if k.startswith("learned/")}


out, (nm, d), refs = sys.argv[1], sys.argv[2].split("="), [a.split("=") for a in sys.argv[3:]]
X = load(d); rows = []
rng = np.random.default_rng(0)
for rn, rd in refs:
    R = load(rd)
    row = dict(run=nm, ref=rn)
    for s in ("val",) + OOD:
        assert len(X[s]) == len(R[s]), (s, len(X[s]), len(R[s]))
        dd = 100 * (X[s] - R[s])
        bs = [dd[rng.integers(0, len(dd), len(dd))].mean() for _ in range(2000)]
        row[s] = f"{100 * X[s].mean():.2f}"
        row[f"d_{s}"] = f"{dd.mean():+.2f} [{np.percentile(bs, 2.5):+.2f}, {np.percentile(bs, 97.5):+.2f}]"
    dd = {s: 100 * (X[s] - R[s]) for s in OOD}
    est = [np.mean([v[rng.integers(0, len(v), len(v))].mean() for v in dd.values()]) for _ in range(2000)]
    row["ood_mean"] = f"{np.mean([100 * X[s].mean() for s in OOD]):.2f}"
    row["ref_ood_mean"] = f"{np.mean([100 * R[s].mean() for s in OOD]):.2f}"
    row["d_ood_mean"] = f"{np.mean([v.mean() for v in dd.values()]):+.2f} [{np.percentile(est, 2.5):+.2f}, {np.percentile(est, 97.5):+.2f}]"
    rows.append(row)
    print(" | ".join(f"{k}={v}" for k, v in row.items()), flush=True)
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)
