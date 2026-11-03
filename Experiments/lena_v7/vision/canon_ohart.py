"""Canonical OH-Art plain LeNA − LoRA: paired image bootstrap (10,000 resamples, numpy seed 0) + exact McNemar."""
import numpy as np, torch
from math import comb
A = torch.load("runs/preds_db_oh_art_lena_s0/preds.pt")["target"].numpy().astype(int)
B = torch.load("runs/preds_db_oh_art_lora_s0/preds.pt")["target"].numpy().astype(int)
n = len(A); d = A - B
rng = np.random.default_rng(0)
idx = rng.integers(0, n, size=(10000, n))
bs = d[idx].mean(1)
lo, hi = np.percentile(bs, [2.5, 97.5])
b01 = int(((A == 1) & (B == 0)).sum()); b10 = int(((A == 0) & (B == 1)).sum()); m = b01 + b10; k = min(b01, b10)
p = min(1.0, 2 * sum(comb(m, i) for i in range(k + 1)) / 2 ** m)
print(f"n {n} LeNA {100*A.mean():.2f} LoRA {100*B.mean():.2f} delta {100*d.mean():+.2f} CI95 [{100*lo:+.4f}, {100*hi:+.4f}] | McNemar LeNA-only {b01} LoRA-only {b10} exact p {p:.4f}")
