"""Camelyon17 stain-shift strength sweep (eval only).

A fixed random subset of the ID validation split (training hospitals 0/3/4; RandomState(0), same subset for every
method) is perturbed in HED space (Tellez et al. style): per image, per channel c in {H, E, D},
hed_c' = hed_c * (1 + sigma * u_c) + sigma * v_c with u, v ~ U(-1, 1) drawn ONCE per image (seeded by the image
index), so all methods and all levels see the same directions and only the strength sigma changes.
Level 0 is the clean image (no HED round trip). The perturbation is applied to the raw 96x96 RGB patch, before
resize / CLIP normalisation. Conversions follow skimage.color.rgb2hed / hed2rgb (skimage is not installed).

  python sweep_stain.py --n 3000 --out runs/sweep_stain
"""
import argparse
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
from camelyon import Classifier, ParquetImages, TARGETS, MEAN, STD  # noqa: E402
from lena_v7 import inject, lena_layers  # noqa: E402

SIGMAS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4]
METHODS = {  # name: (adapter, checkpoint)
    "lora": ("lora", "runs/cam_lora_lr1e-4/best.pt"),
    "lena": ("lena", "runs/cam_lena_lr5e-5/best.pt"),
    "aurora": ("aurora", "runs/cam_aurora_lr1e-4/best.pt"),
    "dora": ("dora", "runs/cam_dora_lr1e-4/best.pt"),
}

# skimage.color.colorconv: rgb_from_hed and its inverse
RGB_FROM_HED = np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]])
HED_FROM_RGB = np.linalg.inv(RGB_FROM_HED)
LOG_ADJ = np.log(1e-6)


def rgb2hed(rgb):
    rgb = np.maximum(rgb, 1e-6)
    return np.maximum((np.log(rgb) / LOG_ADJ) @ HED_FROM_RGB, 0)


def hed2rgb(hed):
    return np.clip(np.exp(-(hed * -LOG_ADJ) @ RGB_FROM_HED), 0, 1)


def stain_perturb(img, sigma, idx, seed=0):
    if sigma == 0:
        return img
    rng = np.random.default_rng([seed, idx])
    u, v = rng.uniform(-1, 1, 3), rng.uniform(-1, 1, 3)
    x = np.asarray(img, dtype=np.float64) / 255.0
    hed = rgb2hed(x) * (1 + sigma * u) + sigma * v
    return Image.fromarray(np.round(hed2rgb(hed) * 255).astype(np.uint8))


class Perturbed(Dataset):
    def __init__(self, base, sigma, tf):
        self.base, self.sigma, self.tf = base, sigma, tf

    def __len__(self):
        return len(self.base.ys)

    def __getitem__(self, i):
        img = Image.open(io.BytesIO(self.base.imgs[i])).convert("RGB")
        return self.tf(stain_perturb(img, self.sigma, i)), self.base.ys[i]


def build(adapter, ckpt, clip):
    model = Classifier(clip)
    for q in model.tower.parameters():
        q.requires_grad_(False)
    kw = dict(r=8, alpha=16.0)
    if adapter == "dora":
        kw["dora"] = True
    if adapter == "lena":
        kw.update(K=16, mode="lena", causal=False, use_checkpoint=False)
    inject(model.tower, TARGETS, adapter={"lena": "lena", "aurora": "aurora"}.get(adapter, "lora"), **kw)
    sd = torch.load(ckpt, map_location="cpu")
    bad = [k for k in sd if k not in model.state_dict()]
    assert not bad, f"checkpoint keys do not match: {bad[:3]}"
    model.load_state_dict(sd, strict=False)
    return model.cuda().eval()


@torch.no_grad()
def predict(model, loader):
    out = []
    for x, y in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(x.cuda(non_blocking=True)).argmax(-1).cpu()
        out.append((pred == y).numpy())
        for L in lena_layers(model):
            L.last = None
    return np.concatenate(out)


def boot_ci(d, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (B, len(d)))
    m = d[idx].mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/mnt/scratch2/users/hchen/data/lena_v7/camelyon17/pkl")
    p.add_argument("--clip", default="openai/clip-vit-base-patch16")
    p.add_argument("--n", type=int, default=3000)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--methods", default="lora,lena,aurora,dora")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(0)
    t0 = time.time()
    base = ParquetImages([os.path.join(a.data, "validation.pkl")], None, keep={0, 3, 4}, limit=a.n)
    print(f"ID val subset {len(base.ys)} (centres {sorted(set(base.cs))}, tumour frac {np.mean(base.ys):.3f}) "
          f"loaded in {time.time() - t0:.0f}s", flush=True)
    tf = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BILINEAR), T.ToTensor(), T.Normalize(MEAN, STD)])
    corr = {}
    for name in a.methods.split(","):
        adapter, ckpt = METHODS[name]
        model = build(adapter, ckpt, a.clip)
        corr[name] = np.zeros((len(SIGMAS), len(base.ys)), dtype=bool)
        for li, s in enumerate(SIGMAS):
            dl = DataLoader(Perturbed(base, s, tf), a.bs, num_workers=a.workers, pin_memory=True)
            corr[name][li] = predict(model, dl)
            print(f"{name} level {li} sigma {s}: acc {corr[name][li].mean():.4f} "
                  f"[{time.time() - t0:.0f}s, mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB]", flush=True)
        del model; torch.cuda.empty_cache()
        np.savez(os.path.join(a.out, "correct.npz"), sigmas=SIGMAS, **corr)
    rows = []
    for li, s in enumerate(SIGMAS):
        r = dict(level=li, sigma=s, n=len(base.ys), **{f"acc_{m}": float(corr[m][li].mean()) for m in corr})
        for other in [m for m in ("lora", "aurora", "dora") if m in corr and "lena" in corr]:
            g, lo, hi = boot_ci(corr["lena"][li].astype(float) - corr[other][li].astype(float))
            r.update({f"gap_lena_{other}": g, f"gap_lena_{other}_lo": lo, f"gap_lena_{other}_hi": hi})
        rows.append(r)
    import csv
    with open(os.path.join(a.out, "stain_sweep.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("DONE", json.dumps(rows), flush=True)


if __name__ == "__main__":
    main()
