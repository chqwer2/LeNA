"""Diabetic-retinopathy grading under cross-site / cross-camera shift: leave-one-source-out over four public fundus
sources (APTOS, DDR, IDRiD, Messidor-2; the obtainable subset of the GDRBench sources), 5-class ICDR grade.
Thin wrapper over domainbed_clip.py (same CLIP ViT-B/16 classifier, adapter placement q,k,v,out,fc1,fc2 of all 12
blocks, r = 8, alpha = 16, zero-shot text head, AdamW + 5 % warmup + cosine, batch 32 drawn uniformly from the pooled
source training images, fixed step budget).

Protocol: for a target source, train on the other three; 20 % of each source is held out (split seed fixed) as
training-domain validation; the checkpoint with the best source-validation accuracy is reported on the whole target
source. Reported per evaluation: accuracy, quadratic-weighted kappa, macro-F1 (5 classes) and the referable-DR AUC
(grade >= 2, score = p(2)+p(3)+p(4)).

Augmentation (--aug geo, default): RandomResizedCrop(224, 0.8-1) + horizontal/vertical flip, no photometric jitter,
so colour / illumination differences between cameras are not trained away (as in camelyon.py).

  python fundus_clip.py --target idrid --adapter lena --lr 5e-5 --out runs/fd_idrid_lena_s0
"""
import argparse
import json
import math
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import lena_v7  # noqa: E402
from lena_v7 import lena_override, adapter_tensors, trainable_count  # noqa: E402
from camelyon import Classifier, MEAN, STD  # noqa: E402
from domainbed_clip import DomainImages, zeroshot_head, build_adapters  # noqa: E402

ROOT = "/mnt/scratch2/users/hchen/data/lena_v7/fundus/pkl"
SOURCES = ["aptos", "ddr", "idrid", "messidor2"]
# zeroshot_head prompts "a photo of a {name}."
NAMES = ["retina with no diabetic retinopathy", "retina with mild diabetic retinopathy",
         "retina with moderate diabetic retinopathy", "retina with severe diabetic retinopathy",
         "retina with proliferative diabetic retinopathy"]


def load_splits(root, target, holdout=0.2, split_seed=0, limit=0):
    tr, va, te = [], [], []
    for di, d in enumerate(SOURCES):
        ims, ys, _ = pickle.load(open(os.path.join(root, f"{d}.pkl"), "rb"))
        items = [(b, y, di) for b, y in zip(ims, ys)]
        idx = np.random.RandomState(split_seed + di).permutation(len(items))
        if limit:
            idx = idx[:limit]
        if d == target:
            te += [items[i] for i in idx]
        else:
            k = int(round(holdout * len(idx)))
            va += [items[i] for i in idx[:k]]; tr += [items[i] for i in idx[k:]]
    return tr, va, te


def metrics(y, p, n=5):
    """accuracy, quadratic-weighted kappa, macro-F1 over grades present in y or pred, referable (>=2) AUC."""
    pred = p.argmax(1)
    acc = float((pred == y).mean())
    O = np.zeros((n, n)); np.add.at(O, (y, pred), 1)
    W = np.array([[(i - j) ** 2 for j in range(n)] for i in range(n)]) / (n - 1) ** 2
    E = np.outer(O.sum(1), O.sum(0)) / max(O.sum(), 1)
    qwk = float(1 - (W * O).sum() / max((W * E).sum(), 1e-12))
    f1s = []
    for c in range(n):
        tp, fp, fn = O[c, c], O[:, c].sum() - O[c, c], O[c, :].sum() - O[c, c]
        if tp + fp + fn > 0:
            f1s.append(2 * tp / (2 * tp + fp + fn))
    yb, s = (y >= 2).astype(int), p[:, 2:].sum(1)
    npos, nneg = yb.sum(), len(yb) - yb.sum()
    if npos and nneg:                                                  # Mann-Whitney U with average ranks
        order = np.argsort(s, kind="mergesort"); r = np.empty(len(s)); r[order] = np.arange(1, len(s) + 1)
        _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
        sums = np.bincount(inv, weights=r); r = (sums / cnt)[inv]
        auc = float((r[yb == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))
    else:
        auc = float("nan")
    return dict(acc=acc, qwk=qwk, f1=float(np.mean(f1s)), auc=auc)


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    P, Y = [], []
    for x, y, _ in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            P.append(model(x.to(dev, non_blocking=True)).float().softmax(-1).cpu()); Y.append(y)
    model.train()
    return metrics(torch.cat(Y).numpy(), torch.cat(P).numpy())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=ROOT)
    p.add_argument("--target", choices=SOURCES, required=True)
    p.add_argument("--clip", default="openai/clip-vit-base-patch16")
    p.add_argument("--adapter", choices=["lena", "lora", "dora", "aurora", "head"], default="lena")
    p.add_argument("--mode", default="lena")
    p.add_argument("--r", type=int, default=8)
    p.add_argument("--alpha", type=float, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--aug", choices=["geo", "domainbed"], default="geo")
    p.add_argument("--holdout", type=float, default=0.2)
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="smoke tests: images per source")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda"

    norm = [T.ToTensor(), T.Normalize(MEAN, STD)]
    if a.aug == "geo":
        tf_tr = T.Compose([T.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.95, 1.05)), T.RandomHorizontalFlip(),
                           T.RandomVerticalFlip()] + norm)
    else:
        tf_tr = T.Compose([T.RandomResizedCrop(224, scale=(0.7, 1.0)), T.RandomHorizontalFlip(),
                           T.ColorJitter(0.3, 0.3, 0.3, 0.3), T.RandomGrayscale(0.1)] + norm)
    tf_te = T.Compose([T.Resize((224, 224))] + norm)
    t0 = time.time()
    tr, va, te = load_splits(a.root, a.target, a.holdout, a.split_seed, a.limit)
    print(dict(train=len(tr), id_val=len(va), target=len(te)), f"loaded in {time.time() - t0:.0f}s", flush=True)
    g = torch.Generator(); g.manual_seed(a.seed)
    dl = DataLoader(DomainImages(tr, tf_tr), a.bs, shuffle=True, num_workers=a.workers, pin_memory=True,
                    drop_last=True, persistent_workers=True, generator=g)
    dle = {k: DataLoader(DomainImages(v, tf_te), 256, num_workers=a.workers, pin_memory=True)
           for k, v in (("id_val", va), ("target", te))}

    model = Classifier(a.clip, n_cls=5)
    zeroshot_head(model, a.clip, NAMES)
    for prm in model.tower.parameters():
        prm.requires_grad_(False)
    adapted = build_adapters(model, a)
    model.to(dev)
    n_tr = trainable_count(model)
    info = dict(vars(a), dataset="fundus", adapted=len(adapted), trainable=n_tr, n_train=len(tr), n_id_val=len(va),
                n_target=len(te), trainable_pct=100 * n_tr / sum(q.numel() for q in model.tower.parameters()))
    print(json.dumps(info), flush=True)

    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd)
    total = a.steps
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))

    def run_eval(step, train_s):
        h = dict(step=step, **{k: evaluate(model, l, dev) for k, l in dle.items()}, train_s=train_s,
                 wall_s=time.time() - t0)
        print(json.dumps(h), flush=True)
        return h

    hist = [run_eval(0, 0.0)]
    best = (-1.0, None)
    step, run, t_train = 0, 0.0, 0.0
    model.train()
    while step < total:
        for x, y, _ in dl:
            t1 = time.time()
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(model(x), y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}")
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            step += 1; run += loss.item(); t_train += time.time() - t1
            if step % 50 == 0:
                print(f"step {step}/{total} loss {run / 50:.4f} {t_train / step:.3f}s/step", flush=True); run = 0.0
            if step % a.eval_every == 0 or step == total:
                hist.append(run_eval(step, t_train))
                if hist[-1]["id_val"]["acc"] > best[0]:
                    best = (hist[-1]["id_val"]["acc"], len(hist) - 1)
                    torch.save(dict(adapter_tensors(model), **{f"head.{k}": v.cpu() for k, v in model.head.state_dict().items()}),
                               os.path.join(a.out, "best.pt"))
            if step >= total:
                break
    sel = hist[best[1]]
    res = dict(info=info, history=hist, selected_step=sel["step"], id_val=sel["id_val"], target=sel["target"],
               target_last=hist[-1]["target"], target_zeroshot=hist[0]["target"],
               s_per_step=t_train / max(step, 1), wall_s=time.time() - t0)
    if a.adapter == "lena":                                          # element checks on the selected checkpoint
        sd = torch.load(os.path.join(a.out, "best.pt"))
        model.load_state_dict({k: v.to(dev) for k, v in sd.items()}, strict=False)
        res["overrides"] = {}
        for c, ov in (("fixed", dict(mode="fixed")), ("plain", dict(mode="plain")), ("id_prior", dict(prior="identity"))):
            with lena_override(**ov):
                res["overrides"][c] = {k: evaluate(model, l, dev) for k, l in dle.items()}
            print("OVERRIDE", c, json.dumps(res["overrides"][c]), flush=True)
    json.dump(res, open(os.path.join(a.out, "result.json"), "w"), indent=1)
    row = dict(target=a.target, adapter=a.adapter, lr=a.lr, seed=a.seed, steps=a.steps, limit=a.limit,
               selected_step=res["selected_step"], **{f"val_{k}": v for k, v in sel["id_val"].items()},
               **{f"tgt_{k}": v for k, v in sel["target"].items()},
               **{f"tgt_last_{k}": v for k, v in res["target_last"].items()},
               **{f"tgt_zs_{k}": v for k, v in res["target_zeroshot"].items()},
               s_per_step=res["s_per_step"], wall_s=res["wall_s"])
    with open(os.path.join(a.out, "summary.csv"), "w") as f:
        f.write(",".join(row) + "\n" + ",".join(str(v) for v in row.values()) + "\n")
    print("RESULT", json.dumps(row))


if __name__ == "__main__":
    main()
