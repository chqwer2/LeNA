"""DomainBed leave-one-domain-out (PACS, Office-Home) with a CLIP ViT-B/16 image tower, a linear head, and
LeNA / LoRA / DoRA / AuroRA adapters. Model, adapter placement and evaluation are imported from camelyon.py
(adapters on q, k, v, out, fc1, fc2 of all 12 blocks, r = 8, alpha = 16; head trained in every variant).

Protocol (Gulrajani & Lopez-Paz, DomainBed): for a target domain, train on the other domains; 20 % of each source
domain is held out (split seed fixed, independent of the run seed) as in-distribution validation, and the checkpoint
with the best source-validation accuracy is reported on the full target domain ("training-domain validation").
DomainBed augmentation (RandomResizedCrop 0.7-1, flip, ColorJitter 0.3, grayscale 0.1). Fixed step budget,
batch 32 drawn uniformly from the pooled source training images, AdamW + 5 % warmup + cosine.

Head: initialised from CLIP zero-shot text embeddings ("a photo of a {class}.") at CLIP's logit scale, so the head
needs no adapter-specific learning rate (--head random gives camelyon.py's random head).

  python domainbed_clip.py --dataset PACS --target sketch --adapter lena --lr 5e-5 --out runs/db_pacs_sketch_lena
"""
import argparse
import io
import json
import math
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import lena_v7  # noqa: E402
from lena_v7 import inject, lena_layers, lena_override, adapter_tensors, trainable_count  # noqa: E402
from camelyon import Classifier, TARGETS, MEAN, STD, evaluate  # noqa: E402

ROOT = "/mnt/scratch2/users/hchen/data/lena_v7/domainbed"
DOMAINS = {"PACS": ["art_painting", "cartoon", "photo", "sketch"],
           "OfficeHome": ["Art", "Clipart", "Product", "Real World"]}


class DomainImages(Dataset):
    """Encoded images of several domains; yields (image, label, domain index)."""

    def __init__(self, items, tf):
        self.items, self.tf = items, tf                               # list of (bytes, label, domain)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        b, y, d = self.items[i]
        return self.tf(Image.open(io.BytesIO(b)).convert("RGB")), y, d


def load_splits(root, dataset, target, holdout=0.2, split_seed=0, limit=0):
    doms = DOMAINS[dataset]
    assert target in doms, f"target {target} not in {doms}"
    tr, va, te = [], [], []
    for di, d in enumerate(doms):
        ims, ys, _ = pickle.load(open(os.path.join(root, dataset, f"{d}.pkl"), "rb"))
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


def classnames(root, dataset):
    names = json.load(open(os.path.join(root, dataset, "meta.json")))["classes"]
    return [n.replace("_", " ").lower() for n in names]


@torch.no_grad()
def zeroshot_head(model, clip_path, names, scale=100.0):
    """Head weight = scale/10 * normalised text embeddings (Classifier multiplies features by 10)."""
    from transformers import CLIPModel, CLIPTokenizer
    clip = CLIPModel.from_pretrained(clip_path).eval()
    tok = CLIPTokenizer.from_pretrained(clip_path)
    t = tok([f"a photo of a {n}." for n in names], padding=True, return_tensors="pt")
    z = F.normalize(clip.get_text_features(**t), dim=-1)
    model.head.weight.copy_(z * scale / 10.0); model.head.bias.zero_()
    del clip


def build_adapters(model, a):
    """Same adapter construction as camelyon.py (lora / dora / aurora / lena)."""
    if a.adapter == "head":
        return []
    kind = {"lena": "lena", "aurora": "aurora"}.get(a.adapter, "lora")
    kw = dict(r=a.r, alpha=a.alpha)
    if a.adapter == "dora":
        kw["dora"] = True
    if kind == "lena":
        kw.update(K=16, mode=a.mode, causal=False, use_checkpoint=False)
    names = inject(model.tower, TARGETS, adapter=kind, **kw)
    return [n for n in names if "encoder.layers" in n]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=ROOT)
    p.add_argument("--dataset", choices=list(DOMAINS), default="PACS")
    p.add_argument("--target", required=True)
    p.add_argument("--clip", default="openai/clip-vit-base-patch16")
    p.add_argument("--adapter", choices=["lena", "lora", "dora", "aurora", "head"], default="lena")
    p.add_argument("--mode", default="lena")
    p.add_argument("--r", type=int, default=8)
    p.add_argument("--alpha", type=float, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--head", choices=["zeroshot", "random"], default="zeroshot")
    p.add_argument("--aug", choices=["domainbed", "flip"], default="domainbed")
    p.add_argument("--holdout", type=float, default=0.2)
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="smoke tests: images per domain")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--triton", action="store_true")
    p.add_argument("--curve_lr_mult", type=float, default=1.0, help="lr factor for LeNA rho/A2/kappa_raw (omega has --omega_lr_mult)")
    p.add_argument("--omega_lr", type=float, default=0.0, help="absolute lr for omega (overrides --omega_lr_mult: mult = omega_lr / lr)")
    p.add_argument("--omega_lr_mult", type=float, default=1.0, help="H18: lr factor for the per-layer Stein scale omega only (no decay)")
    p.add_argument("--eval_preds", default=None, help="eval-only: best.pt of a finished run (same adapter args); per-image correctness of id_val/target -> preds.pt in --out, then exit")
    p.add_argument("--neff", action="store_true", help="R1: Stein noise variance x design effect 1+(N-1)rho_tok")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    if a.omega_lr > 0:
        a.omega_lr_mult = a.omega_lr / a.lr
    a.target = {d.replace(" ", "_"): d for d in DOMAINS[a.dataset]}.get(a.target, a.target)   # "Real_World"
    lena_v7.set_compile(a.compile)
    lena_v7.set_triton(a.triton)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda"

    norm = [T.ToTensor(), T.Normalize(MEAN, STD)]
    if a.aug == "domainbed":
        tf_tr = T.Compose([T.RandomResizedCrop(224, scale=(0.7, 1.0)), T.RandomHorizontalFlip(),
                           T.ColorJitter(0.3, 0.3, 0.3, 0.3), T.RandomGrayscale(0.1)] + norm)
    else:
        tf_tr = T.Compose([T.Resize((224, 224)), T.RandomHorizontalFlip()] + norm)
    tf_te = T.Compose([T.Resize((224, 224))] + norm)                 # DomainBed test transform
    t0 = time.time()
    tr, va, te = load_splits(a.root, a.dataset, a.target, a.holdout, a.split_seed, a.limit)
    names = classnames(a.root, a.dataset)
    print(dict(train=len(tr), id_val=len(va), target=len(te), classes=len(names)), f"loaded in {time.time() - t0:.0f}s",
          flush=True)
    g = torch.Generator(); g.manual_seed(a.seed)
    dl = DataLoader(DomainImages(tr, tf_tr), a.bs, shuffle=True, num_workers=a.workers, pin_memory=True,
                    drop_last=True, persistent_workers=True, generator=g)
    dle = {k: DataLoader(DomainImages(v, tf_te), 256, num_workers=a.workers, pin_memory=True)
           for k, v in (("id_val", va), ("target", te))}

    model = Classifier(a.clip, n_cls=len(names))
    if a.head == "zeroshot":
        zeroshot_head(model, a.clip, names)
    for prm in model.tower.parameters():
        prm.requires_grad_(False)
    adapted = build_adapters(model, a)
    if a.neff:                                                       # R1 runtime patch (lena_mri_ext)
        import atexit
        import lena_mri_ext
        lena_mri_ext.install(model, neff=True)
        lena_mri_ext.EXT["acc_eval"] = {}

        def _dump_s():
            acc = lena_mri_ext.EXT["acc_eval"]
            st = {k: acc[k] / max(acc.get("n", 1), 1) for k in ("s", "bend") if k in acc}
            st["n_fits"] = acc.get("n", 0)
            print("STEIN_EVAL", json.dumps(st), flush=True)
            json.dump(st, open(os.path.join(a.out, "stein_eval.json"), "w"))
        atexit.register(_dump_s)
    model.to(dev)
    n_tr = trainable_count(model)
    info = dict(vars(a), adapted=len(adapted), trainable=n_tr, n_train=len(tr), n_id_val=len(va), n_target=len(te),
                trainable_pct=100 * n_tr / sum(q.numel() for q in model.tower.parameters()))
    print(json.dumps(info), flush=True)
    if a.eval_preds:
        sd = torch.load(a.eval_preds, map_location="cpu")
        assert not [k for k in sd if k not in model.state_dict()], "checkpoint keys do not match the model"
        model.load_state_dict({k: v.to(dev) for k, v in sd.items()}, strict=False)
        model.eval(); out = {}
        with torch.no_grad():
            for k, l in dle.items():
                ok = []
                for x, y, _ in l:
                    x, y = x.to(dev, non_blocking=True), y.to(dev)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        ok.append((model(x).argmax(-1) == y).cpu())
                out[k] = torch.cat(ok)
        os.makedirs(a.out, exist_ok=True)
        torch.save(dict(ckpt=a.eval_preds, **out), os.path.join(a.out, "preds.pt"))
        print("EVAL_PREDS", json.dumps({k: float(v.float().mean()) for k, v in out.items()}), flush=True)
        return

    params = [q for q in model.parameters() if q.requires_grad]
    om = [q for n_, q in model.named_parameters() if q.requires_grad and n_.endswith(".omega")] if a.omega_lr_mult != 1.0 else []
    omid = {id(q) for q in om}                                       # H18: omega in its own group (no decay)
    cur = [q for n_, q in model.named_parameters() if q.requires_grad and id(q) not in omid
           and n_.rsplit(".", 1)[-1] in ("rho", "A2", "kappa_raw")] if a.curve_lr_mult != 1.0 else []
    cid = {id(q) for q in cur}
    groups = [dict(params=[q for q in params if id(q) not in omid and id(q) not in cid])] + (
        [dict(params=cur, lr=a.lr * a.curve_lr_mult)] if cur else []) + (
        [dict(params=om, lr=a.lr * a.omega_lr_mult, weight_decay=0.0)] if om else [])
    print("PARAM_GROUPS", [dict(n=sum(q.numel() for q in g["params"]), lr=g.get("lr", a.lr)) for g in groups], flush=True)
    opt = torch.optim.AdamW(groups, lr=a.lr, weight_decay=a.wd)
    total = a.steps
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))

    def run_eval(step, train_s):
        accs = {k: evaluate(model, l, dev)[0] for k, l in dle.items()}
        h = dict(step=step, **accs, train_s=train_s, wall_s=time.time() - t0)
        print(json.dumps(h), flush=True)
        return h

    hist = [run_eval(0, 0.0)]                                        # zero-shot head + identity adapters
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
                if hist[-1]["id_val"] > best[0]:
                    best = (hist[-1]["id_val"], len(hist) - 1)
                    torch.save(dict(adapter_tensors(model), **{f"head.{k}": v.cpu() for k, v in model.head.state_dict().items()}),
                               os.path.join(a.out, "best.pt"))
                model.train()
            if step >= total:
                break
    sel = hist[best[1]]
    res = dict(info=info, history=hist, selected_step=sel["step"], id_val=sel["id_val"], target=sel["target"],
               target_last=hist[-1]["target"], id_val_last=hist[-1]["id_val"], target_zeroshot=hist[0]["target"],
               s_per_step=t_train / max(step, 1), wall_s=time.time() - t0)
    if a.adapter == "lena":                                          # element checks on the selected checkpoint
        sd = torch.load(os.path.join(a.out, "best.pt"))
        model.load_state_dict({k: v.to(dev) for k, v in sd.items()}, strict=False)
        res["overrides"] = {}
        for c, ov in (("learned", {}), ("fixed", dict(mode="fixed")), ("plain", dict(mode="plain")),
                      ("id_prior", dict(prior="identity"))):
            with lena_override(**ov):
                res["overrides"][c] = {k: evaluate(model, l, dev)[0] for k, l in dle.items()}
            print("OVERRIDE", c, json.dumps(res["overrides"][c]), flush=True)
    json.dump(res, open(os.path.join(a.out, "result.json"), "w"), indent=1)
    keys = ("dataset", "target", "adapter", "lr", "seed", "steps")
    row = dict({k: info[k] for k in keys}, selected_step=res["selected_step"], id_val=res["id_val"],
               target=res["target"], target_last=res["target_last"], target_zeroshot=res["target_zeroshot"],
               s_per_step=res["s_per_step"], wall_s=res["wall_s"])
    with open(os.path.join(a.out, "summary.csv"), "w") as f:
        f.write(",".join(row) + "\n" + ",".join(str(v) for v in row.values()) + "\n")
    print("RESULT", json.dumps(row))


if __name__ == "__main__":
    main()
