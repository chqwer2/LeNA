"""CLIP few-shot diagnostics for the element and motivation figures.

Part 1 (LeNA checkpoint): per test set (ImageNet, V2, Sketch, A, R), n images: fitted ordinates theta, step s, kept knots
p, fit statistic F per vision module and channel, histograms of the first read u, and (u, v) of the first n_uv images
for split-half reliability of the fitted curve.
Part 2 (LoRA checkpoint, method-agnostic probe): put a curve on LoRA's code (identity start = LoRA exactly; LeNA layer
in "fixed" mode with A1 <- A, B <- B) and train only the curves, separately on each test set's labelled images;
save the curve learned for each set.

  python diag_clip.py --lena runs/clip_lena_s1/adapter.pt --lora runs/clip_lora_s1/adapter.pt --out runs/diag_clip
"""
import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from lena_v7 import inject, LeNALinear, lena_layers  # noqa: E402
from clip_fewshot import CLIPClassifier, Images, classnames, MEAN, STD, PKL  # noqa: E402
from torchvision import transforms as T  # noqa: E402

SETS = ["val", "v2", "sketch", "a", "r"]
TF = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor(), T.Normalize(MEAN, STD)])


def build(kind, ckpt, mode="lena"):
    m = CLIPClassifier("openai/clip-vit-base-patch16", classnames())
    for q in m.parameters():
        q.requires_grad_(False)
    kw = dict(r=2, alpha=math.sqrt(2), dropout=0.0, K=16, causal=False, use_checkpoint=False, mode=mode)
    inject(m.clip, ["q_proj", "k_proj", "v_proj"], adapter="lena", **kw)
    sd = torch.load(ckpt, map_location="cpu")
    if kind == "lora":                                        # LoRA tensors -> fixed-curve LeNA layers (identity prior = LoRA)
        sd = {k: v for k, v in sd.items() if k.endswith(".A1") or k.endswith(".B")}
    missing = [k for k in sd if k not in m.state_dict()]
    assert not missing, missing[:3]
    m.load_state_dict(sd, strict=False)
    return m.cuda()


def vision_layers(m):
    return [(n, L) for n, L in m.named_modules() if isinstance(L, LeNALinear) and "vision_model" in n]


@torch.no_grad()
def part1(m, sets, n, n_uv, out):
    m.eval(); vis = vision_layers(m); res = {"names": [n_ for n_, _ in vis]}
    for s in sets:
        ds = Images(os.path.join(PKL, f"{s}.pkl"), TF, limit=n, seed=0)
        rec = {k: [] for k in ("theta", "s", "F", "p", "uhist", "u", "v")}
        for x, _ in DataLoader(ds, 50, num_workers=4):
            for _, L in vis:
                L.log_stats = True
            with torch.autocast("cuda", dtype=torch.float16):
                m.image_features(x.cuda())
            for k in ("theta", "s", "F", "p"):
                rec[k].append(torch.stack([L.last[k].float().cpu() for _, L in vis], 1))
            u = torch.stack([L.last["u"].float() for _, L in vis], 1)   # (B, M, N, r)
            rec["uhist"].append(torch.stack([torch.histc(u[b].clamp(-6, 6), 60, -6, 6) for b in range(u.shape[0])]).cpu())
            if sum(len(t) for t in rec["u"]) < n_uv:
                rec["u"].append(u.half().cpu()); rec["v"].append(torch.stack([L.last["v"].float() for _, L in vis], 1).half().cpu())
            for _, L in vis:
                L.log_stats = False; L.last = None
        res[s] = {k: torch.cat(v) for k, v in rec.items()}
        print("part1", s, {k: tuple(v.shape) for k, v in res[s].items()}, flush=True)
    res["rho"] = torch.stack([L.rho.detach().float().cpu() for _, L in vis])
    torch.save(res, os.path.join(out, "diag_clip_part1.pt"))


def part2(ckpt, sets, n_train, steps, out, halves=False):
    curves = {}
    jobs = [(s, h) for s in sets for h in ((0, 1) if halves else (None,))]
    for s, h in jobs:
        torch.manual_seed(0 if h is None else 10 + h)
        m = build("lora", ckpt, mode="fixed")
        rhos = [L.rho for L in lena_layers(m)]
        for r_ in rhos:
            r_.requires_grad_(True)
        ds = Images(os.path.join(PKL, f"{s}.pkl"), TF, limit=n_train * (2 if halves else 1), seed=1)
        if halves:                                            # two disjoint halves of the same set: the noise floor
            ds.imgs, ds.ys = ds.imgs[h * n_train:(h + 1) * n_train], ds.ys[h * n_train:(h + 1) * n_train]
        cls = sorted(set(ds.ys)) if s in ("a", "r") else None
        dl = DataLoader(ds, 32, shuffle=True, num_workers=4, drop_last=True)
        opt = torch.optim.AdamW(rhos, lr=1e-2, weight_decay=0.0)
        step = 0; m.train()
        while step < steps:
            for x, y in dl:
                x, y = x.cuda(), y.cuda()
                with torch.autocast("cuda", dtype=torch.float16):
                    tf = m.text_features()
                    if cls is not None:
                        keep = torch.tensor(cls, device=x.device); tf = tf[keep]
                        remap = torch.full((1000,), -1, dtype=torch.long, device=x.device); remap[keep] = torch.arange(len(keep), device=x.device); y = remap[y]
                    loss = F.cross_entropy(100.0 * m.image_features(x).float() @ tf.float().t(), y)
                opt.zero_grad(); loss.backward(); opt.step(); step += 1
                if step >= steps:
                    break
        names = [n_ for n_, L in m.named_modules() if isinstance(L, LeNALinear)]
        curves[s if h is None else f"{s}_half{h}"] = dict(names=names, rho=torch.stack([r_.detach().float().cpu() for r_ in rhos]), loss=float(loss))
        print("part2", s, h, "final loss", float(loss), flush=True)
    torch.save(curves, os.path.join(out, "diag_clip_probe_halves.pt" if halves else "diag_clip_probe.pt"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lena", required=True); p.add_argument("--lora", required=True); p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=300); p.add_argument("--n_uv", type=int, default=100)
    p.add_argument("--n_train", type=int, default=1000); p.add_argument("--steps", type=int, default=300)
    p.add_argument("--skip1", action="store_true"); p.add_argument("--skip2", action="store_true")
    p.add_argument("--halves", action="store_true", help="part 2 twice per set on disjoint halves (noise floor)")
    a = p.parse_args(); os.makedirs(a.out, exist_ok=True)
    if not a.skip1:
        part1(build("lena", a.lena), SETS, a.n, a.n_uv, a.out)
    if not a.skip2:
        part2(a.lora, SETS, a.n_train, a.steps, a.out, halves=a.halves)
    print("DONE")


if __name__ == "__main__":
    main()
