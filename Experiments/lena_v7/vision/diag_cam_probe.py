"""Method-agnostic probe on Camelyon17: take the trained LoRA model, put a curve on its code (identity start = LoRA exactly;
LeNA layer in "fixed" mode with A1 <- A, B <- B) and train only the curves, separately on each hospital's labelled
patches. Saves the curve learned for each hospital and the accuracy before and after on held-out patches of that hospital.

  python diag_cam_probe.py --lora runs/cam_lora_lr1e-4/best.pt --out runs/diag_cam
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from lena_v7 import inject, LeNALinear, lena_layers  # noqa: E402
from camelyon import Classifier, ParquetImages, TARGETS, MEAN, STD  # noqa: E402
from torchvision import transforms as T  # noqa: E402

D = "/mnt/scratch2/users/hchen/data/lena_v7/camelyon17/pkl"
TF = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BILINEAR), T.ToTensor(), T.Normalize(MEAN, STD)])


def build(ckpt):
    m = Classifier("openai/clip-vit-base-patch16")
    for q in m.tower.parameters():
        q.requires_grad_(False)
    inject(m.tower, TARGETS, adapter="lena", r=8, alpha=16, K=16, mode="fixed", causal=False, use_checkpoint=False)
    sd = torch.load(ckpt, map_location="cpu")
    sd = {k: v for k, v in sd.items() if k.endswith(".A1") or k.endswith(".B") or k.startswith("head.")}
    assert not [k for k in sd if k not in m.state_dict()]
    m.load_state_dict(sd, strict=False)
    for q in m.head.parameters():
        q.requires_grad_(False)
    return m.cuda()


@torch.no_grad()
def acc(m, ds, idx):
    m.eval(); ok = 0
    for i in range(0, len(idx), 256):
        b = [ds[k] for k in idx[i:i + 256]]
        x = torch.stack([t[0] for t in b]).cuda(); y = torch.tensor([t[1] for t in b]).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ok += (m(x).argmax(-1) == y).sum().item()
    return ok / len(idx)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lora", required=True); p.add_argument("--out", required=True)
    p.add_argument("--n_train", type=int, default=1000); p.add_argument("--n_test", type=int, default=1000)
    p.add_argument("--steps", type=int, default=300)
    a = p.parse_args(); os.makedirs(a.out, exist_ok=True)
    data = {"validation": ParquetImages([os.path.join(D, "validation.pkl")], TF), "test": ParquetImages([os.path.join(D, "test.pkl")], TF)}
    res = {}
    for h, split in ((0, "validation"), (3, "validation"), (4, "validation"), (1, "validation"), (2, "test")):
        ds = data[split]; rng = np.random.RandomState(h)
        idx = rng.permutation([i for i, c in enumerate(ds.cs) if c == h]); tr, te = list(idx[:a.n_train]), list(idx[a.n_train:a.n_train + a.n_test])
        torch.manual_seed(0); m = build(a.lora); rhos = [L.rho for L in lena_layers(m)]
        for r_ in rhos:
            r_.requires_grad_(True)
        before = acc(m, ds, te)
        opt = torch.optim.AdamW(rhos, lr=1e-2, weight_decay=0.0); step = 0; m.train()
        dl = DataLoader(Subset(ds, tr), 32, shuffle=True, num_workers=4, drop_last=True)
        while step < a.steps:
            for x, y, _ in dl:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = F.cross_entropy(m(x.cuda()).float(), y.cuda())
                opt.zero_grad(); loss.backward(); opt.step(); step += 1
                if step >= a.steps:
                    break
        after = acc(m, ds, te)
        names = [n for n, L in m.named_modules() if isinstance(L, LeNALinear)]
        res[h] = dict(names=names, rho=torch.stack([r_.detach().float().cpu() for r_ in rhos]), acc_before=before, acc_after=after)
        print(f"hospital {h}: held-out acc LoRA {before:.4f} -> with its own curve {after:.4f}", flush=True)
    torch.save(res, os.path.join(a.out, "diag_cam_probe.pt")); print("DONE")


if __name__ == "__main__":
    main()
