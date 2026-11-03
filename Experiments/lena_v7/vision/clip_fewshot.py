"""Few-shot CLIP ViT-B/16 on ImageNet, tested on ImageNet and its shifted variants (V2 matched-frequency, Sketch,
A, R), with LeNA / LoRA adapters in the same code base.

Protocol of CLIP-LoRA (Zanella & Ben Ayed, CVPRW 2024; github.com/MaxZanella/CLIP-LoRA, lora.py / main.py):
adapters on q, k, v of all 12 blocks of both encoders, r = 2, alpha = 1 with the alpha / sqrt(r) scale, dropout 0.25,
AdamW (lr 2e-4, wd 1e-2), cosine to 1e-6, batch 32, 500 x shots iterations, fp16 autocast, logit scale 100,
"a photo of a {}." with the CLIP-LoRA ImageNet class names, RandomResizedCrop(224, 0.08-1, bicubic) + flip; the
last iterate is tested. Few-shot images are sampled per class with the run seed (CLIP-LoRA does the same).
ImageNet-A and -R are scored over their 200 classes.

LeNA statistics are bidirectional within each image (197 tokens) and within each prompt (its valid tokens).

  python clip_fewshot.py --adapter lena --train imagenet_16shot_seed1.pkl --out runs/clip_lena
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
from transformers import CLIPModel, CLIPTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import lena_v7  # noqa: E402
from lena_v7 import (inject, lena_context, lena_layers, lena_override, adapter_tensors,  # noqa: E402
                     trainable_count)

MEAN, STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
PKL = "/mnt/scratch2/users/hchen/data/lena_v7/imagenet_pkl"
EVAL_SETS = ["val", "v2", "sketch", "a", "r"]


def classnames(path="/mnt/scratch2/users/hchen/CLIP-LoRA/datasets/imagenet.py"):
    """The 1000 ImageNet class names of CLIP-LoRA (datasets/imagenet.py, `imagenet_classes`)."""
    src = open(path).read()
    a = src.index("imagenet_classes = ["); b = src.index("]", a)
    names = eval(src[a + len("imagenet_classes = "):b + 1])      # a literal list of strings
    assert len(names) == 1000
    return names


class Images(Dataset):
    def __init__(self, path, tf, limit=0, seed=0):
        imgs, ys, _ = pickle.load(open(path, "rb"))
        if limit and limit < len(ys):
            idx = np.random.RandomState(seed).permutation(len(ys))[:limit]
            imgs, ys = [imgs[i] for i in idx], [ys[i] for i in idx]
        self.imgs, self.ys, self.tf = imgs, ys, tf

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, i):
        return self.tf(Image.open(io.BytesIO(self.imgs[i])).convert("RGB")), self.ys[i]


class CLIPClassifier(torch.nn.Module):
    def __init__(self, path, names):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(path)
        tok = CLIPTokenizer.from_pretrained(path)
        t = tok([f"a photo of a {n.replace('_', ' ')}." for n in names], padding=True, return_tensors="pt")
        self.register_buffer("ids", t["input_ids"]); self.register_buffer("mask", t["attention_mask"])

    def text_features(self):
        with lena_context(attention_mask=self.mask, causal=False):
            z = self.clip.get_text_features(input_ids=self.ids, attention_mask=self.mask)
        return F.normalize(z, dim=-1)

    def image_features(self, x):
        with lena_context(attention_mask=None, causal=False):
            return F.normalize(self.clip.get_image_features(pixel_values=x), dim=-1)


PREDS = {"buf": None}


@torch.no_grad()
def evaluate(model, loader, classes=None, log=0):
    model.eval()
    with torch.autocast("cuda", dtype=torch.float16):
        tf = model.text_features().float()
    if classes is not None:
        keep = torch.tensor(classes, device=tf.device)
        remap = torch.full((1000,), -1, dtype=torch.long, device=tf.device); remap[keep] = torch.arange(len(keep),
                                                                                                     device=tf.device)
        tf = tf[keep]
    vis = [L for n, L in model.named_modules() if L in set(lena_layers(model)) and "vision_model" in n]
    correct = n = 0
    rec = dict(s=[], F=[], correct=[])
    for x, y in loader:
        x, y = x.cuda(non_blocking=True), y.cuda()
        lg = log and n < log and vis
        for L in vis:
            L.log_stats = bool(lg)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model.image_features(x).float() @ tf.t()
        yy = remap[y] if classes is not None else y
        ok = logits.argmax(-1) == yy
        correct += ok.sum().item(); n += len(y)
        if PREDS["buf"] is not None:
            PREDS["buf"].append(ok.cpu())
        if lg:
            rec["s"].append(torch.stack([L.last["s"] for L in vis], 1).half().cpu())        # (B, M_vis, r)
            rec["F"].append(torch.stack([L.last["F"] for L in vis], 1).float().cpu())
            rec["correct"].append(ok.cpu())
        for L in vis:
            L.log_stats = False; L.last = None
    model.train()
    return correct / max(n, 1), ({k: torch.cat(v) for k, v in rec.items()} if rec["s"] else None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clip", default="openai/clip-vit-base-patch16")
    p.add_argument("--train", required=True, help="pickle (image bytes, labels, _) of the few-shot training set")
    p.add_argument("--eval_dir", default=PKL)
    p.add_argument("--eval_sets", default=",".join(EVAL_SETS))
    p.add_argument("--adapter", choices=["lena", "lena_aurora", "lena_dora", "lora", "dora", "aurora", "zeroshot"], default="lena")
    p.add_argument("--mode", default="lena", help="LeNA training mode: lena | fixed (s=0) | plain (s=1)")
    p.add_argument("--r", type=int, default=2)
    p.add_argument("--alpha", type=float, default=1.0, help="CLIP-LoRA alpha; the scale is alpha / sqrt(r)")
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--shots", type=int, default=16)
    p.add_argument("--n_iters", type=int, default=500)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--limit_eval", type=int, default=0)
    p.add_argument("--max_steps", type=int, default=0)
    p.add_argument("--log_stats", type=int, default=2000, help="images per eval set with logged LeNA statistics")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--load", default="", help="adapter.pt of a finished run: skip training, evaluate only")
    p.add_argument("--stat_drop", type=float, default=0.0, help="LeNA: token thinning of the statistics in training")
    p.add_argument("--prior_lr_mult", type=float, default=1.0, help="LeNA: learning-rate factor for rho and omega (no decay)")
    p.add_argument("--compile", action="store_true", help="torch.compile the LeNA fit path (same outputs)")
    p.add_argument("--triton", action="store_true", help="fused Triton kernels for the mode-lena fit (lena_triton.py)")
    p.add_argument("--omega_lr", type=float, default=0.0, help="absolute lr for omega (overrides --omega_lr_mult: mult = omega_lr / lr)")
    p.add_argument("--omega_lr_mult", type=float, default=1.0, help="H18: lr factor for the per-layer Stein scale omega only (no decay)")
    p.add_argument("--save_preds", action="store_true", help="save per-image correctness of every eval set -> preds.pt")
    p.add_argument("--only_learned", action="store_true", help="evaluate the learned pass only (skip fixed/plain/id_prior)")
    p.add_argument("--neff", action="store_true", help="R1: Stein noise variance x design effect 1+(N-1)rho_tok")
    p.add_argument("--qt", action="store_true", help="R3: lena_qt quantile-transport layer")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    if a.omega_lr > 0:
        a.omega_lr_mult = a.omega_lr / a.lr
    lena_v7.set_compile(a.compile)
    lena_v7.set_triton(a.triton)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)

    tf_tr = T.Compose([T.RandomResizedCrop(224, scale=(0.08, 1), interpolation=T.InterpolationMode.BICUBIC),
                       T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(MEAN, STD)])
    tf_te = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor(),
                       T.Normalize(MEAN, STD)])
    model = CLIPClassifier(a.clip, classnames())
    for q in model.parameters():
        q.requires_grad_(False)
    names = []
    if a.adapter != "zeroshot":
        kw = dict(r=a.r, alpha=a.alpha * math.sqrt(a.r), dropout=a.dropout)  # alpha_ours / r = alpha / sqrt(r)
        if a.adapter in ("lena", "lena_aurora", "lena_dora"):
            kw.update(K=16, causal=False, use_checkpoint=False, mode=a.mode)
        if a.adapter == "lena_aurora":
            kw["base_nl"] = "aurora"
        if a.adapter == "lena_dora":
            kw["dora"] = True
        if a.adapter == "dora":
            kw["dora"] = True
        kind = {"dora": "lora", "lena_aurora": "lena", "lena_dora": "lena"}.get(a.adapter, a.adapter)
        if a.adapter == "lena" and a.qt:
            import lena_qt
            lena_qt.register()
            kind = "lena_qt"
        names = inject(model.clip, ["q_proj", "k_proj", "v_proj"], adapter=kind, **kw)
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
    model.cuda()
    n_tr = trainable_count(model)
    info = dict(vars(a), adapted=len(names), trainable=n_tr)
    print(json.dumps(info), flush=True)

    sets = {s: Images(os.path.join(a.eval_dir, f"{s}.pkl"), tf_te, limit=a.limit_eval) for s in a.eval_sets.split(",")}
    sub = {s: sorted(set(d.ys)) for s, d in sets.items()}
    loaders = {s: DataLoader(d, 256, num_workers=a.workers, pin_memory=True) for s, d in sets.items()}
    res = dict(info=info)
    t0 = time.time()
    if a.load:
        sd = torch.load(a.load, map_location="cpu")
        missing = [k for k in sd if k not in model.state_dict()]
        assert not missing, f"checkpoint keys do not match the model: {missing[:3]}"
        model.load_state_dict(sd, strict=False)
    elif a.adapter != "zeroshot":
        tr = Images(a.train, tf_tr)
        dl = DataLoader(tr, a.bs, shuffle=True, num_workers=a.workers, pin_memory=True, drop_last=False,
                        persistent_workers=True)
        total = a.max_steps or a.n_iters * a.shots
        for L in lena_layers(model):
            L.stat_drop = a.stat_drop
        pri = [q for n, q in model.named_parameters() if q.requires_grad and (n.endswith(".rho") or n.endswith(".omega") or n.endswith(".kappa_raw"))]
        rest = [q for n, q in model.named_parameters() if q.requires_grad and not (n.endswith(".rho") or n.endswith(".omega") or n.endswith(".kappa_raw"))]
        om = []
        if a.omega_lr_mult != 1.0:                                   # H18: omega in its own group, rho untouched
            om = [q for n, q in model.named_parameters() if q.requires_grad and n.endswith(".omega")]
            ids = {id(q) for q in om}
            pri = [q for q in pri if id(q) not in ids]
        groups = [dict(params=rest)] + ([dict(params=pri, lr=a.lr * a.prior_lr_mult, weight_decay=0.0)] if pri else []) \
            + ([dict(params=om, lr=a.lr * a.omega_lr_mult, weight_decay=0.0)] if om else [])
        print("PARAM_GROUPS", [dict(n=sum(q.numel() for q in g["params"]), lr=g.get("lr", a.lr)) for g in groups], flush=True)
        opt = torch.optim.AdamW(groups, lr=a.lr, weight_decay=1e-2, betas=(0.9, 0.999))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, total, eta_min=1e-6)
        scaler = torch.amp.GradScaler()
        step, hist = 0, []
        model.train()
        while step < total:
            for x, y in dl:
                x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = 100.0 * model.image_features(x) @ model.text_features().t()
                    loss = F.cross_entropy(logits.float(), y)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
                step += 1
                if step % 100 == 0:
                    hist.append(dict(step=step, loss=loss.item(), s_per_step=(time.time() - t0) / step))
                    print(json.dumps(hist[-1]), flush=True)
                if step >= total:
                    break
        res["history"] = hist
        torch.save(adapter_tensors(model), os.path.join(a.out, "adapter.pt"))
    conds = [("learned", {})]
    if a.adapter in ("lena", "lena_aurora", "lena_dora") and not a.only_learned:
        conds += [("fixed", dict(mode="fixed")), ("plain", dict(mode="plain")), ("id_prior", dict(prior="identity"))]
    stats, preds = {}, {}
    for c, ov in conds:
        res[c] = {}
        with lena_override(**ov):
            for s, l in loaders.items():
                PREDS["buf"] = [] if a.save_preds else None
                acc, rec = evaluate(model, l, classes=sub[s] if s in ("a", "r") else None,
                                    log=a.log_stats if (c == "learned" and a.adapter in ("lena", "lena_aurora", "lena_dora")) else 0)
                res[c][s] = acc
                if a.save_preds:
                    preds[f"{c}/{s}"] = torch.cat(PREDS["buf"]); PREDS["buf"] = None
                if rec is not None:
                    stats[s] = rec
        res[c]["ood_avg"] = float(np.mean([res[c][s] for s in ("v2", "sketch", "a", "r") if s in res[c]]))
        print("EVAL", c, json.dumps(res[c]), f"[{time.time() - t0:.0f}s]", flush=True)
    if stats:
        torch.save(stats, os.path.join(a.out, "stats.pt"))
    if preds:
        torch.save(preds, os.path.join(a.out, "preds.pt"))
    json.dump(res, open(os.path.join(a.out, "result.json"), "w"), indent=1)
    print("RESULT", json.dumps(res["learned"]))


if __name__ == "__main__":
    main()
