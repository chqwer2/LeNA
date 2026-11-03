"""CLIP few-shot ImageNet corruption sweep (eval only).

The ImageNet val subset of the CLIP runs (val.pkl, RandomState(0) permutation[:2000], identical to --limit_eval 2000)
after the test transform's Resize(224)+CenterCrop(224), corrupted on the fly with ImageNet-C parameters
(Hendrycks & Dietterich 2019, imagenet_c/corruptions.py) at severities 1..5; severity 0 is the clean image.
gaussian_noise c = .08/.12/.18/.26/.38; gaussian_blur sigma = 1/2/3/4/6; contrast c = .4/.3/.2/.1/.05;
jpeg quality = 25/18/15/10/7. Noise is seeded per image index (identical images for every method).

  python sweep_corrupt.py --n 2000 --out runs/sweep_corrupt
"""
import argparse
import csv
import io
import json
import math
import os
import sys
import time

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from clip_fewshot import CLIPClassifier, Images, classnames, MEAN, STD, PKL  # noqa: E402
from lena_v7 import inject, lena_context, lena_layers  # noqa: E402

METHODS = {"lora": "runs/clip_lora_s1/adapter.pt", "lena": "runs/clip_lena_s1/adapter.pt",
           "aurora": "runs/clip_aurora_s1/adapter.pt", "dora": "runs/clip_dora_s1/adapter.pt"}
CORR = ["gaussian_noise", "gaussian_blur", "contrast", "jpeg"]
PARAMS = {"gaussian_noise": [.08, .12, .18, .26, .38], "gaussian_blur": [1, 2, 3, 4, 6],
          "contrast": [.4, .3, .2, .1, .05], "jpeg": [25, 18, 15, 10, 7]}


def corrupt(img, kind, sev, idx):
    if sev == 0:
        return img
    c = PARAMS[kind][sev - 1]
    if kind == "gaussian_blur":
        return img.filter(ImageFilter.GaussianBlur(radius=c))
    if kind == "jpeg":
        buf = io.BytesIO(); img.save(buf, "JPEG", quality=c); buf.seek(0)
        return Image.open(buf).convert("RGB")
    x = np.asarray(img, dtype=np.float64) / 255.0
    if kind == "gaussian_noise":
        x = x + np.random.default_rng([0, idx, sev]).normal(size=x.shape, scale=c)
    elif kind == "contrast":
        m = x.mean(axis=(0, 1), keepdims=True)
        x = (x - m) * c + m
    return Image.fromarray(np.round(np.clip(x, 0, 1) * 255).astype(np.uint8))


class Corrupted(Dataset):
    def __init__(self, base, kind, sev):
        self.base, self.kind, self.sev = base, kind, sev
        self.pre = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224)])
        self.post = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.base.ys)

    def __getitem__(self, i):
        img = self.pre(Image.open(io.BytesIO(self.base.imgs[i])).convert("RGB"))
        return self.post(corrupt(img, self.kind, self.sev, i)), self.base.ys[i]


def build(name, clip):
    model = CLIPClassifier(clip, classnames())
    for q in model.parameters():
        q.requires_grad_(False)
    kw = dict(r=2, alpha=1.0 * math.sqrt(2), dropout=0.25)
    if name == "lena":
        kw.update(K=16, causal=False, use_checkpoint=False, mode="lena")
    if name == "dora":
        kw["dora"] = True
    inject(model.clip, ["q_proj", "k_proj", "v_proj"], adapter={"dora": "lora"}.get(name, name), **kw)
    sd = torch.load(METHODS[name], map_location="cpu")
    bad = [k for k in sd if k not in model.state_dict()]
    assert not bad, f"checkpoint keys do not match: {bad[:3]}"
    model.load_state_dict(sd, strict=False)
    return model.cuda().eval()


@torch.no_grad()
def predict(model, tf, loader):
    out = []
    for x, y in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model.image_features(x.cuda(non_blocking=True)).float() @ tf.t()
        out.append((logits.argmax(-1).cpu() == y).numpy())
        for L in lena_layers(model):
            L.last = None
    return np.concatenate(out)


def boot_ci(d, B=1000, seed=0):
    idx = np.random.default_rng(seed).integers(0, len(d), (B, len(d)))
    m = d[idx].mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clip", default="openai/clip-vit-base-patch16")
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--methods", default="lora,lena,aurora,dora")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(0)
    t0 = time.time()
    base = Images(os.path.join(PKL, "val.pkl"), None, limit=a.n, seed=0)
    print(f"val subset {len(base.ys)} loaded in {time.time() - t0:.0f}s", flush=True)
    conds = [("clean", 0)] + [(k, s) for k in CORR for s in range(1, 6)]
    corr = {}
    for name in a.methods.split(","):
        model = build(name, a.clip)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            tf = model.text_features().float()
        corr[name] = {}
        for k, s in conds:
            dl = DataLoader(Corrupted(base, k, s), a.bs, num_workers=a.workers, pin_memory=True)
            corr[name][(k, s)] = predict(model, tf, dl)
            print(f"{name} {k} s{s}: acc {corr[name][(k, s)].mean():.4f} "
                  f"[{time.time() - t0:.0f}s, mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB]", flush=True)
        del model; torch.cuda.empty_cache()
        np.savez(os.path.join(a.out, "correct.npz"),
                 **{f"{m}__{k}__{s}": v for m, d in corr.items() for (k, s), v in d.items()})
    ms = list(corr)
    rows = []

    def row(kind, s, get):
        r = dict(corruption=kind, severity=s, n=len(base.ys), **{f"acc_{m}": float(get(m).mean()) for m in ms})
        if "lena" in ms:
            for o in [m for m in ("lora", "aurora", "dora") if m in ms]:
                g, lo, hi = boot_ci(get("lena") - get(o))
                r.update({f"gap_lena_{o}": g, f"gap_lena_{o}_lo": lo, f"gap_lena_{o}_hi": hi})
        rows.append(r)

    row("clean", 0, lambda m: corr[m][("clean", 0)].astype(float))
    for k in CORR:
        for s in range(1, 6):
            row(k, s, lambda m: corr[m][(k, s)].astype(float))
    for s in range(1, 6):   # per-image mean over the 4 corruptions, then paired bootstrap
        row("mean", s, lambda m: np.mean([corr[m][(k, s)] for k in CORR], 0))
    with open(os.path.join(a.out, "corrupt_sweep.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("DONE", json.dumps(rows), flush=True)


if __name__ == "__main__":
    main()
