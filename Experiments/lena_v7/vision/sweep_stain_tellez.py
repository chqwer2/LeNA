"""Camelyon17 stain-shift sweep with the HED jitter of Tellez et al. (2018, IEEE TMI, App. B, Eq. 5-7).

S = -log(P + eps) @ M^-1 (raw optical density, natural log, P in [0, 1]); S'_c = alpha_c * S_c + beta_c with
alpha_c ~ U(1 - sigma, 1 + sigma), beta_c ~ U(-sigma, sigma); P' = exp(-S' @ M) - eps. Tellez use sigma = 0.05
("HED-light"); Tellez et al. 2019 (MedIA) also use 0.2 ("HED-strong").

sweep_stain.py applied the same alpha/beta form to skimage>=0.19's rgb2hed, which divides the OD by -log(1e-6)
(about 13.8); its additive term is therefore about 13.8 times larger in OD units than Tellez's at equal sigma.

Same evaluation as sweep_stain.py: the same RandomState(0) subset of the ID validation split (centres 0/3/4), one
random direction (u, v ~ U(-1, 1)) per image seeded by the image index and shared by all levels and methods,
paired bootstrap CIs. Modes:
  python sweep_stain_tellez.py --thumbs out.npz        # 8 patches x levels, for choosing the grid
  python sweep_stain_tellez.py --stats out.csv         # per-patch mean HED OD for hospitals and jitter levels
  python sweep_stain_tellez.py --out runs/sweep_stain_tellez
"""
import argparse
import csv
import io
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from camelyon import ParquetImages, MEAN, STD  # noqa: E402
from sweep_stain import METHODS, RGB_FROM_HED, HED_FROM_RGB, build, predict, boot_ci  # noqa: E402

SIGMAS = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
EPS = 1.0 / 255


def rgb2od(x):
    return -np.log(x + EPS) @ HED_FROM_RGB


def od2rgb(s):
    return np.clip(np.exp(-s @ RGB_FROM_HED) - EPS, 0, 1)


def tellez(img, sigma, idx, seed=0):
    if sigma == 0:
        return img
    rng = np.random.default_rng([seed, idx])
    u, v = rng.uniform(-1, 1, 3), rng.uniform(-1, 1, 3)
    x = np.asarray(img, dtype=np.float64) / 255.0
    s = rgb2od(x) * (1 + sigma * u) + sigma * v
    return Image.fromarray(np.round(od2rgb(s) * 255).astype(np.uint8))


class Perturbed(Dataset):
    def __init__(self, base, sigma, tf):
        self.base, self.sigma, self.tf = base, sigma, tf

    def __len__(self):
        return len(self.base.ys)

    def __getitem__(self, i):
        img = Image.open(io.BytesIO(self.base.imgs[i])).convert("RGB")
        return self.tf(tellez(img, self.sigma, i)), self.base.ys[i]


def mean_od(img):
    """Per-patch mean H, E, D optical density over tissue pixels (mean RGB OD > 0.15; all pixels if none)."""
    x = np.asarray(img, dtype=np.float64) / 255.0
    od = -np.log(x + EPS)
    tissue = od.mean(-1) > 0.15
    s = (od @ HED_FROM_RGB)[tissue] if tissue.any() else (od @ HED_FROM_RGB).reshape(-1, 3)
    return s.mean(0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/mnt/scratch2/users/hchen/data/lena_v7/camelyon17/pkl")
    p.add_argument("--clip", default="openai/clip-vit-base-patch16")
    p.add_argument("--n", type=int, default=3000)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--methods", default="lora,lena,aurora,dora")
    p.add_argument("--sigmas", default=",".join(map(str, SIGMAS)))
    p.add_argument("--thumbs", default="")
    p.add_argument("--stats", default="")
    p.add_argument("--out", default="")
    a = p.parse_args()
    sigmas = [float(s) for s in a.sigmas.split(",")]
    t0 = time.time()
    base = ParquetImages([os.path.join(a.data, "validation.pkl")], None, keep={0, 3, 4}, limit=a.n)
    print(f"ID val subset {len(base.ys)} centres {sorted(set(base.cs))} [{time.time() - t0:.0f}s]", flush=True)

    if a.thumbs:
        idx = list(range(8))
        arr = np.stack([[np.asarray(tellez(Image.open(io.BytesIO(base.imgs[i])).convert("RGB"), s, i)) for s in sigmas]
                        for i in idx])
        np.savez(a.thumbs, thumbs=arr, sigmas=sigmas, labels=[base.ys[i] for i in idx], idx=idx)
        print("THUMBS", arr.shape, flush=True)
        return

    if a.stats:
        rows = []
        for i in range(len(base.ys)):                   # jitter levels on the ID subset
            img = Image.open(io.BytesIO(base.imgs[i])).convert("RGB")
            for s in sigmas:
                rows.append(dict(source="jitter", sigma=s, centre=base.cs[i], idx=i, y=base.ys[i],
                                 **dict(zip("HED", mean_od(tellez(img, s, i))))))
        for split, keep in (("validation.pkl", {1}), ("test.pkl", {2}), ("validation.pkl", {0, 3, 4})):
            ds = ParquetImages([os.path.join(a.data, split)], None, keep=keep, limit=a.n)
            for i in range(len(ds.ys)):
                img = Image.open(io.BytesIO(ds.imgs[i])).convert("RGB")
                rows.append(dict(source=f"hospital_{split.split('.')[0]}", sigma="", centre=ds.cs[i], idx=i,
                                 y=ds.ys[i], **dict(zip("HED", mean_od(img)))))
            print(f"stats {split} {sorted(keep)} n={len(ds.ys)} [{time.time() - t0:.0f}s]", flush=True)
        with open(a.stats, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        print("STATS_DONE", len(rows), flush=True)
        return

    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(0)
    tf = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BILINEAR), T.ToTensor(), T.Normalize(MEAN, STD)])
    corr = {}
    for name in a.methods.split(","):
        adapter, ckpt = METHODS[name]
        model = build(adapter, ckpt, a.clip)
        corr[name] = np.zeros((len(sigmas), len(base.ys)), dtype=bool)
        for li, s in enumerate(sigmas):
            dl = DataLoader(Perturbed(base, s, tf), a.bs, num_workers=a.workers, pin_memory=True)
            corr[name][li] = predict(model, dl)
            print(f"{name} level {li} sigma {s}: acc {corr[name][li].mean():.4f} "
                  f"[{time.time() - t0:.0f}s, mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB]", flush=True)
        del model; torch.cuda.empty_cache()
        np.savez(os.path.join(a.out, "correct.npz"), sigmas=sigmas, **corr)
    rows = []
    for li, s in enumerate(sigmas):
        r = dict(level=li, sigma=s, n=len(base.ys), **{f"acc_{m}": float(corr[m][li].mean()) for m in corr})
        for other in [m for m in ("lora", "aurora", "dora") if m in corr and "lena" in corr]:
            g, lo, hi = boot_ci(corr["lena"][li].astype(float) - corr[other][li].astype(float))
            r.update({f"gap_lena_{other}": g, f"gap_lena_{other}_lo": lo, f"gap_lena_{other}_hi": hi})
        rows.append(r)
    with open(os.path.join(a.out, "stain_sweep_tellez.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("DONE", json.dumps(rows), flush=True)


if __name__ == "__main__":
    main()
