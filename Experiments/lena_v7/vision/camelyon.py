"""WILDS Camelyon17 (tumour vs normal patches; train hospitals 0/3/4, OOD val hospital 1, OOD test hospital 2) with
a CLIP ViT-B/16 image tower, a linear head, and LeNA / LoRA adapters in the same code base.

Protocol after Kumar et al. 2022 (CLIP ViT-B/16 on Camelyon17): 224 x 224 bilinear resize, horizontal flip only,
3 epochs, cosine schedule, AdamW, model selection on ID validation. Adapters on q, k, v, out, fc1, fc2 of all 12
blocks; the head is trained in every variant. LeNA statistics are bidirectional over each image's 197 tokens
(one fitted curve per image and channel).

  python camelyon.py --adapter lena --lr 3e-4 --out runs/cam_lena_lr3e-4
"""
import argparse
import glob
import contextlib
import io
import json
import math
import os
import sys
import time

import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from transformers import CLIPVisionModelWithProjection

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import lena_v7  # noqa: E402
from lena_v7 import (inject, lena_layers, lena_override, adapter_tensors, trainable_count,  # noqa: E402
                     matched_lora_rank)

MEAN, STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)   # CLIP
TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]


class ParquetImages(Dataset):
    """All rows of a split (pickle of (png bytes, label, center) converted from the WILDS parquet files, see
    slurm/l7cam_pkl.sh) kept as encoded bytes; decoded in the workers."""

    def __init__(self, files, tf, keep=None, limit=0):
        imgs, ys, cs = [], [], []
        for f in files:
            im, lab, cen = pickle.load(open(f, "rb"))
            for i, c in enumerate(cen):
                if keep is None or c in keep:
                    imgs.append(im[i]); ys.append(lab[i]); cs.append(c)
        if limit:
            rng = np.random.RandomState(0); idx = rng.permutation(len(ys))[:limit]
            imgs, ys, cs = [imgs[i] for i in idx], [ys[i] for i in idx], [cs[i] for i in idx]
        self.imgs, self.ys, self.cs, self.tf = imgs, ys, cs, tf

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, i):
        return self.tf(Image.open(io.BytesIO(self.imgs[i])).convert("RGB")), self.ys[i], self.cs[i]


class Classifier(nn.Module):
    def __init__(self, path, n_cls=2):
        super().__init__()
        self.tower = CLIPVisionModelWithProjection.from_pretrained(path)
        self.head = nn.Linear(self.tower.config.projection_dim, n_cls)

    def forward(self, x):
        z = self.tower(pixel_values=x).image_embeds
        return self.head(F.normalize(z, dim=-1) * 10.0)


def d_pairs(model):
    return [(m.in_features, m.out_features) for n, m in model.tower.named_modules()
            if isinstance(m, nn.Linear) and n.split(".")[-1] in TARGETS and "encoder.layers" in n]


POOL = {"on": False, "chunk": 64}                                  # --pool_batch: LeNA fit pooled per hospital chunk


class HospitalBatches(torch.utils.data.Sampler):
    """--hospital_batches: every batch from a single training hospital (chosen in proportion to its size)."""

    def __init__(self, centers, bs, seed=0):
        self.by = {}
        for i, c in enumerate(centers):
            self.by.setdefault(int(c), []).append(i)
        self.keys = sorted(self.by)
        self.p = np.array([len(self.by[k]) for k in self.keys], float); self.p /= self.p.sum()
        self.bs, self.n, self.rng = bs, len(centers) // bs, np.random.RandomState(seed)

    def __len__(self):
        return self.n

    def __iter__(self):
        for _ in range(self.n):
            k = self.keys[self.rng.choice(len(self.keys), p=self.p)]
            yield self.rng.choice(self.by[k], self.bs, replace=False).tolist()


def pool_ctx(c, dev, chunk):
    if not POOL["on"]:
        return contextlib.nullcontext()
    import lena_mri_ext
    c = torch.as_tensor(c)
    key = c.long() * 100000 + torch.arange(len(c)) // chunk          # same hospital and same chunk of the batch
    return lena_mri_ext.groups(torch.unique(key, return_inverse=True)[1].to(dev))


PREDS = {"buf": None}


@torch.no_grad()
def evaluate(model, loader, dev, log_stats=False, max_logged=4000):
    model.eval()
    layers = lena_layers(model)
    correct = n = 0
    rec = dict(s=[], F=[], center=[], y=[], correct=[])
    for x, y, c in loader:
        x, y = x.to(dev, non_blocking=True), y.to(dev)
        lg = log_stats and len(rec["y"]) * loader.batch_size < max_logged and layers
        for L in layers:
            L.log_stats = bool(lg)
        with torch.autocast("cuda", dtype=torch.bfloat16), pool_ctx(c, dev, POOL["chunk"]):
            pred = model(x).argmax(-1)
        correct += (pred == y).sum().item(); n += len(y)
        if PREDS["buf"] is not None:
            PREDS["buf"].append((pred == y).cpu())
        if lg:
            rec["s"].append(torch.stack([L.last["s"] for L in layers], 1).half().cpu())       # (B, M, r)
            rec["F"].append(torch.stack([L.last["F"] for L in layers], 1).float().cpu())
            rec["center"].append(c); rec["y"].append(y.cpu()); rec["correct"].append((pred == y).cpu())
        for L in layers:
            L.log_stats = False; L.last = None
    model.train()
    if rec["y"]:
        rec = {k: torch.cat(v) for k, v in rec.items()}
    return correct / max(n, 1), (rec if rec["y"] is not None and len(rec["y"]) else None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/mnt/scratch2/users/hchen/data/lena_v7/camelyon17/pkl")
    p.add_argument("--clip", default="openai/clip-vit-base-patch16")
    p.add_argument("--adapter", choices=["lena", "lora", "lora_dagger", "dora", "aurora", "head"], default="lena")
    p.add_argument("--mode", default="lena")
    p.add_argument("--r", type=int, default=8)
    p.add_argument("--alpha", type=float, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit_train", type=int, default=0, help="smoke tests")
    p.add_argument("--limit_eval", type=int, default=0)
    p.add_argument("--max_steps", type=int, default=0)
    p.add_argument("--gate_drop", type=float, default=0.0, help="LeNA: per-input probability of training with the shared curve only (s=0)")
    p.add_argument("--save_steps", default="", help="comma list of steps at which adapter+head are saved (training dynamics)")
    p.add_argument("--deff_scale", type=float, default=1.0, help="R1 eval knob: deff = 1 + scale*(N-1)rho_tok (1 = as trained, 0 = no n_eff)")
    p.add_argument("--omega_lr_mult", type=float, default=1.0, help="H18: lr factor for the per-layer Stein scale omega only (no decay)")
    p.add_argument("--save_preds", action="store_true", help="with --eval_only: per-patch correctness of every split -> preds.pt")
    p.add_argument("--eval_only", default="", help="best.pt of a finished run (same adapter args): evaluate it "
                   "on the (possibly --limit_eval subsampled) splits and write eval_only.json in --out")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--neff", action="store_true", help="R1: Stein noise variance x design effect 1+(N-1)rho_tok")
    p.add_argument("--qt", action="store_true", help="R3: lena_qt quantile-transport layer")
    p.add_argument("--ttr", action="store_true", help="R12: lena_ttr layer (cross-fit, separate query, CV shrinkage)")
    p.add_argument("--curve_lr_mult", type=float, default=1.0, help="lr multiplier for LeNA rho/omega/A2")
    p.add_argument("--freeze_A2", action="store_true", help="LeNA: keep the second read A2 at its random init")
    p.add_argument("--hospital_batches", action="store_true", help="each training batch from one hospital")
    p.add_argument("--pool_batch", action="store_true", help="LeNA fit pooled over the batch (train) / 64-patch "
                   "same-hospital chunks (eval); implies --hospital_batches")
    p.add_argument("--grad_clip", type=float, default=0.0, help="clip total grad norm (0 = off); logs per-step grad norms")
    p.add_argument("--stop_epochs", type=int, default=0, help="stop after this many epochs, schedule unchanged (0 = off)")
    p.add_argument("--compile", action="store_true", help="torch.compile the LeNA fit path (same outputs)")
    p.add_argument("--triton", action="store_true", help="fused Triton kernels for the mode-lena fit (lena_triton.py)")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    lena_v7.set_compile(a.compile)
    lena_v7.set_triton(a.triton)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda"

    tf_tr = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BILINEAR), T.RandomHorizontalFlip(),
                       T.ToTensor(), T.Normalize(MEAN, STD)])
    tf_te = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BILINEAR), T.ToTensor(),
                       T.Normalize(MEAN, STD)])
    g = lambda s: [os.path.join(a.data, f"{s}.pkl")]  # noqa: E731
    t0 = time.time()
    tr = ParquetImages(g("train"), tf_tr, limit=a.limit_train)
    ev = {"id_val": ParquetImages(g("validation"), tf_te, keep={0, 3, 4}, limit=a.limit_eval),
          "ood_val": ParquetImages(g("validation"), tf_te, keep={1}, limit=a.limit_eval),
          "ood_test": ParquetImages(g("test"), tf_te, limit=a.limit_eval)}
    print({k: len(v) for k, v in dict(train=tr, **ev).items()}, f"loaded in {time.time() - t0:.0f}s", flush=True)
    if a.hospital_batches or a.pool_batch:
        dl = DataLoader(tr, batch_sampler=HospitalBatches(tr.cs, a.bs, a.seed), num_workers=a.workers, pin_memory=True,
                        persistent_workers=True)
    else:
        dl = DataLoader(tr, a.bs, shuffle=True, num_workers=a.workers, pin_memory=True, drop_last=True,
                        persistent_workers=True)
    dle = {k: DataLoader(v, 256, num_workers=a.workers, pin_memory=True) for k, v in ev.items()}

    model = Classifier(a.clip)
    for prm in model.tower.parameters():
        prm.requires_grad_(False)
    if a.adapter != "head":
        r = a.r
        if a.adapter == "lora_dagger":
            r = matched_lora_rank(d_pairs(model), a.r, 16)
        kind = {"lena": "lena", "aurora": "aurora"}.get(a.adapter, "lora")
        kw = dict(r=r, alpha=a.alpha * r / a.r)                     # same alpha/r scale across ranks
        if a.adapter == "dora":
            kw["dora"] = True
        if kind == "lena":
            kw.update(K=16, mode=a.mode, causal=False, use_checkpoint=False)
            if a.freeze_A2:
                kw["freeze_A2"] = True
        if kind == "lena" and a.qt:
            import lena_qt
            lena_qt.register()
            kind = "lena_qt"
        if kind == "lena" and a.ttr:
            import lena_ttr
            lena_ttr.register()
            kind = "lena_ttr"
        names = inject(model.tower, TARGETS, adapter=kind, **kw)
        names = [n for n in names if "encoder.layers" in n]
        for L in lena_layers(model):
            L.gate_drop = a.gate_drop
    else:
        names, r = [], 0
    if a.pool_batch:
        import lena_mri_ext
        lena_mri_ext.install(model)
        POOL["on"] = True
    if a.neff:                                                       # R1 runtime patch (lena_mri_ext)
        import atexit
        import lena_mri_ext
        lena_mri_ext.install(model, neff=True)
        lena_mri_ext.EXT["acc_eval"] = {}
        lena_mri_ext.EXT["deff_scale"] = a.deff_scale

        def _dump_s():
            acc = lena_mri_ext.EXT["acc_eval"]
            st = {k: acc[k] / max(acc.get("n", 1), 1) for k in ("s", "bend") if k in acc}
            st["n_fits"] = acc.get("n", 0)
            print("STEIN_EVAL", json.dumps(st), flush=True)
            json.dump(st, open(os.path.join(a.out, "stein_eval.json"), "w"))
        atexit.register(_dump_s)
    model.to(dev)
    n_tr = trainable_count(model)
    info = dict(vars(a), rank=r, adapted=len(names), trainable=n_tr,
                trainable_pct=100 * n_tr / sum(q.numel() for q in model.tower.parameters()))
    print(json.dumps(info), flush=True)

    if a.eval_only:
        sd = torch.load(a.eval_only, map_location="cpu")
        model.load_state_dict(sd, strict=False)
        assert not [k for k in sd if k not in model.state_dict()], "checkpoint keys do not match the model"
        res, preds = {}, {}
        for k, l in dle.items():
            PREDS["buf"] = [] if a.save_preds else None
            res[k] = evaluate(model, l, dev)[0]
            if a.save_preds:
                preds[k] = torch.cat(PREDS["buf"]); PREDS["buf"] = None
        if preds:
            torch.save(dict(ckpt=a.eval_only, limit_eval=a.limit_eval, **preds), os.path.join(a.out, "preds.pt"))
        json.dump(dict(ckpt=a.eval_only, limit_eval=a.limit_eval, **res), open(os.path.join(a.out, "eval_only.json"), "w"))
        print("EVAL_ONLY", json.dumps(res))
        return
    params = [q for q in model.parameters() if q.requires_grad]
    om = [q for n_, q in model.named_parameters() if q.requires_grad and n_.endswith(".omega")] if a.omega_lr_mult != 1.0 else []
    omid = {id(q) for q in om}                                       # H18: omega in its own group (no decay)
    params = [q for q in params if id(q) not in omid]
    if a.curve_lr_mult != 1.0:
        cur = [q for n_, q in model.named_parameters() if q.requires_grad and id(q) not in omid
               and n_.rsplit(".", 1)[-1] in ("rho", "omega", "A2", "kappa_raw")]
        ids = {id(q) for q in cur}
        groups = [dict(params=[q for q in params if id(q) not in ids]), dict(params=cur, lr=a.lr * a.curve_lr_mult)]
    else:
        groups = [dict(params=params)]
    if om:
        groups.append(dict(params=om, lr=a.lr * a.omega_lr_mult, weight_decay=0.0))
    print("PARAM_GROUPS", [dict(n=sum(q.numel() for q in g["params"]), lr=g.get("lr", a.lr)) for g in groups], flush=True)
    opt = torch.optim.AdamW(groups, lr=a.lr, weight_decay=a.wd)
    total = a.max_steps or a.epochs * len(dl)
    warm = max(1, int(0.05 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    hist, step, best = [], 0, (-1, None)
    gnorms = []
    model.train()
    for ep in range(a.epochs):
        t1, run = time.time(), 0.0
        for x, y, _ in dl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16), pool_ctx(torch.zeros(len(y)), dev, len(y)):
                loss = F.cross_entropy(model(x), y)
            opt.zero_grad(set_to_none=True); loss.backward()
            if a.grad_clip > 0:
                gnorms.append(float(torch.nn.utils.clip_grad_norm_(params, a.grad_clip)))
            opt.step(); sched.step()
            step += 1; run += loss.item()
            if a.save_steps and step in {int(t) for t in a.save_steps.split(",")}:
                torch.save(dict(adapter_tensors(model), **{f"head.{k}": v.cpu() for k, v in model.head.state_dict().items()}),
                           os.path.join(a.out, f"ckpt_{step}.pt"))
            if step % 200 == 0:
                print(f"ep {ep} step {step}/{total} loss {run / 200:.4f} {(time.time() - t1) / (step - ep * len(dl)):.3f}s/step"
                      + (f" gnorm {np.mean(gnorms[-200:]):.3f} max {np.max(gnorms[-200:]):.3f}" if gnorms else ""), flush=True); run = 0.0
            if step >= total:
                break
        accs, recs = {}, {}
        for k, l in dle.items():
            accs[k], recs[k] = evaluate(model, l, dev, log_stats=a.adapter == "lena")
        hist.append(dict(epoch=ep, step=step, **accs, train_s=time.time() - t1))
        print(json.dumps(hist[-1]), flush=True)
        om_ = [q.detach().float() for n_, q in model.named_parameters() if n_.endswith(".omega")]
        if om_:
            t_ = torch.stack(om_)
            print("OMEGA_STATS", json.dumps(dict(epoch=hist[-1].get("epoch"), mean=float(t_.mean()), min=float(t_.min()), max=float(t_.max()), absmean=float(t_.abs().mean()))), flush=True)
        if accs["id_val"] > best[0]:
            best = (accs["id_val"], ep)
            torch.save(dict(adapter_tensors(model), **{f"head.{k}": v.cpu() for k, v in model.head.state_dict().items()}),
                       os.path.join(a.out, "best.pt"))
            if a.adapter == "lena":
                torch.save(recs, os.path.join(a.out, "best_stats.pt"))
        if gnorms:
            json.dump(gnorms, open(os.path.join(a.out, "grad_norms.json"), "w"))
        if step >= total or (a.stop_epochs and ep + 1 >= a.stop_epochs):
            break
    sel = hist[best[1]]
    res = dict(info=info, history=hist, selected_epoch=best[1], id_val=sel["id_val"], ood_val=sel["ood_val"],
               ood_test=sel["ood_test"])
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
    print("RESULT", json.dumps({k: res[k] for k in ("selected_epoch", "id_val", "ood_val", "ood_test")}))


if __name__ == "__main__":
    main()
