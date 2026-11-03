"""Test-time adaptation baselines on WILDS Camelyon17 for the LoRA / LeNA checkpoints of camelyon.py.

Methods (one stream per evaluation split, model reset before each split, stream = the same 10k random subset as
camelyon.py --limit_eval 10000, shuffled once with seed 0, one pass):
  source      no adaptation (must reproduce eval_only.json)
  tent_ln     TENT (Wang et al. 2021): online entropy minimisation of the predictions, updating only the LayerNorm
              affine parameters of the image tower; SGD momentum 0.9; prediction for a batch = logits of the
              forward pass that is then used for the update (as in the TENT reference code)
  tent_adapt  same objective, updating the adapter parameters (LoRA A1/B, or every trainable LeNA tensor)
  feat_stats  closest no-backward analogue of BN-statistics adaptation for a LayerNorm ViT: each test batch's
              pre-head embedding is re-standardised per dimension to the source mean/std (estimated label-free
              on held-out ID-validation images that are not in the 10k ID-val eval subset)

  python tta_camelyon.py --adapter lora --ckpt runs/cam_lora_lr1e-4/best.pt --method tent_ln --lr 1e-3 \
      --out_csv runs/tta/lora_tent_ln_1e-3_bs64.csv
"""
import argparse
import copy
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as T

from camelyon import Classifier, ParquetImages, TARGETS, MEAN, STD
import lena_v7
from lena_v7 import inject, lena_layers, LeNALinear, LoRALinear


def build(a, dev):
    model = Classifier(a.clip)
    for p in model.parameters():
        p.requires_grad_(False)
    kind = "lena" if a.adapter == "lena" else "lora"
    kw = dict(r=a.r, alpha=a.alpha)
    if kind == "lena":
        kw.update(K=16, mode="lena", causal=False, use_checkpoint=False)
    inject(model.tower, TARGETS, adapter=kind, **kw)
    sd = torch.load(a.ckpt, map_location="cpu")
    missing = [k for k in sd if k not in model.state_dict()]
    assert not missing, f"checkpoint keys do not match the model: {missing[:3]}"
    model.load_state_dict(sd, strict=False)
    return model.to(dev).eval()


def select_params(model, method):
    for p in model.parameters():
        p.requires_grad_(False)
    ps = []
    if method == "tent_ln":
        for m in model.tower.modules():
            if isinstance(m, nn.LayerNorm):
                ps += [m.weight, m.bias]
    elif method == "tent_adapt":
        for m in model.tower.modules():
            if isinstance(m, (LeNALinear, LoRALinear)):
                ps += [p for n, p in m.named_parameters() if not n.startswith("base.") and p.dtype.is_floating_point]
    for p in ps:
        p.requires_grad_(True)
    return ps


def embed(model, x):
    return F.normalize(model.tower(pixel_values=x).image_embeds, dim=-1) * 10.0


@torch.no_grad()
def source_feat_stats(model, loader, dev):
    zs = []
    for x, _, _ in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            zs.append(embed(model, x.to(dev)).float())
    z = torch.cat(zs)
    return z.mean(0), z.std(0) + 1e-5


def run_stream(model, method, loader, dev, opt=None, src=None):
    correct = n = 0
    ent_sum, gpu = 0.0, 0.0
    torch.cuda.synchronize(); t0 = time.time()
    for x, y, _ in loader:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        torch.cuda.synchronize(); tb = time.time()
        if method in ("tent_ln", "tent_adapt"):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
            lf = logits.float()
            ent = -(lf.softmax(-1) * lf.log_softmax(-1)).sum(-1).mean()
            opt.zero_grad(set_to_none=True); ent.backward(); opt.step()
            pred = lf.detach().argmax(-1); ent_sum += ent.item() * len(y)
        else:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                if method == "feat_stats":
                    z = embed(model, x).float()
                    mu, sd = (z.mean(0), z.std(0) + 1e-5) if len(z) > 1 else src
                    z = (z - mu) / sd * src[1] + src[0]
                    logits = model.head(z)
                else:
                    logits = model(x)
            pred = logits.float().argmax(-1)
        torch.cuda.synchronize(); gpu += time.time() - tb
        correct += (pred == y).sum().item(); n += len(y)
    torch.cuda.synchronize()
    return correct / n, n, time.time() - t0, gpu, ent_sum / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/mnt/scratch2/users/hchen/data/lena_v7/camelyon17/pkl")
    p.add_argument("--clip", default="openai/clip-vit-base-patch16")
    p.add_argument("--adapter", choices=["lena", "lora"], required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--r", type=int, default=8)
    p.add_argument("--alpha", type=float, default=16)
    p.add_argument("--method", choices=["source", "tent_ln", "tent_adapt", "feat_stats"], required=True)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--limit_eval", type=int, default=10000)
    p.add_argument("--limit_stream", type=int, default=0, help="smoke test: first N images of the shuffled stream")
    p.add_argument("--n_src", type=int, default=5000, help="feat_stats: held-out ID-val images for source stats")
    p.add_argument("--splits", default="id_val,ood_val,ood_test")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--triton", action="store_true", help="fused Triton kernels for the LeNA fit (same outputs)")
    p.add_argument("--out_csv", required=True)
    a = p.parse_args()
    torch.manual_seed(0); np.random.seed(0)
    lena_v7.set_triton(a.triton)
    dev = "cuda"
    tf = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BILINEAR), T.ToTensor(), T.Normalize(MEAN, STD)])
    pk = lambda s: [os.path.join(a.data, f"{s}.pkl")]  # noqa: E731
    spec = {"id_val": ("validation", {0, 3, 4}), "ood_val": ("validation", {1}), "ood_test": ("test", None)}

    model = build(a, dev)
    init = copy.deepcopy(model.state_dict())
    src = None
    if a.method == "feat_stats":                     # label-free source stats on ID-val images outside the eval subset
        full = ParquetImages(pk("validation"), tf, keep={0, 3, 4})
        used = set(np.random.RandomState(0).permutation(len(full))[:a.limit_eval].tolist())
        rest = [i for i in range(len(full)) if i not in used]
        rest = np.random.RandomState(1).permutation(rest)[:a.n_src].tolist()
        src = source_feat_stats(model, DataLoader(Subset(full, rest), 256, num_workers=a.workers), dev)
        print(f"source stats from {len(rest)} held-out ID-val images", flush=True)
        del full

    os.makedirs(os.path.dirname(os.path.abspath(a.out_csv)), exist_ok=True)
    rows = []
    for split in a.splits.split(","):
        f, keep = spec[split]
        ds = ParquetImages(pk(f), tf, keep=keep, limit=a.limit_eval)
        order = np.random.RandomState(0).permutation(len(ds))
        if a.limit_stream:
            order = order[:a.limit_stream]
        dl = DataLoader(Subset(ds, order.tolist()), a.bs, shuffle=False, num_workers=a.workers, pin_memory=True)
        model.load_state_dict(init)                  # reset per split
        ps = select_params(model, a.method)
        opt = torch.optim.SGD(ps, lr=a.lr, momentum=a.momentum) if ps else None
        torch.cuda.reset_peak_memory_stats()
        acc, n, sec, gpu, ent = run_stream(model, a.method, dl, dev, opt, src)
        row = dict(adapter=a.adapter + ("+triton" if a.triton else ""), method=a.method, lr=a.lr if ps else "", bs=a.bs, split=split, acc=round(acc, 4),
                   n=n, sec=round(sec, 1), sec_per_1k=round(1000 * sec / n, 2),
                   gpu_sec_per_1k=round(1000 * gpu / n, 2),
                   peak_mem_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                   n_updated=sum(q.numel() for q in ps), mean_entropy=round(ent, 4) if ps else "",
                   needs_backward=int(bool(ps)), needs_batch=int(a.method != "source"))
        print("ROW", row, flush=True)
        rows.append(row)
        del ds, dl
    with open(a.out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print("DONE", a.out_csv, flush=True)


if __name__ == "__main__":
    main()
