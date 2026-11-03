"""Diagnostics of a Camelyon17 LeNA checkpoint for the figures: per-image fitted curves, steps and fit statistics
per hospital, per-patch maps of the per-image correction, layer knock-out (s = 0 in one block), and the LoRA update
map of a LoRA checkpoint for comparison.

  python diag_camelyon.py --lena runs/cam_lena_lr5e-5/best.pt --lora runs/cam_lora_lr1e-4/best.pt --out runs/diag_cam
"""
import argparse
import io
import json
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import lena_v7  # noqa: E402
from lena_v7 import inject, lena_layers, LeNALinear, LoRALinear, hat_index, spline_eval  # noqa: E402
from camelyon import Classifier, ParquetImages, TARGETS, MEAN, STD  # noqa: E402
from torchvision import transforms as T  # noqa: E402

D = "/mnt/scratch2/users/hchen/data/lena_v7/camelyon17/pkl"
HOSP = [0, 3, 4, 1, 2]


def build(kind, ckpt, dev):
    m = Classifier("openai/clip-vit-base-patch16")
    for q in m.tower.parameters():
        q.requires_grad_(False)
    kw = dict(r=8, alpha=16)
    if kind == "lena":
        kw.update(K=16, mode="lena", causal=False, use_checkpoint=False)
    inject(m.tower, TARGETS, adapter=kind, **kw)
    m.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
    return m.to(dev).eval()


def subset(per_h, n_per, seed=0):
    tf = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BILINEAR), T.ToTensor(), T.Normalize(MEAN, STD)])
    out = {}
    for split, hs in (("validation", (0, 3, 4, 1)), ("test", (2,))):
        ds = ParquetImages([os.path.join(D, f"{split}.pkl")], tf)
        rng = np.random.RandomState(seed)
        for h in hs:
            idx = [i for i, c in enumerate(ds.cs) if c == h]
            pos = [i for i in idx if ds.ys[i] == 1]; neg = [i for i in idx if ds.ys[i] == 0]
            pick = list(rng.choice(pos, n_per // 2, replace=False)) + list(rng.choice(neg, n_per // 2, replace=False))
            out[h] = (ds, pick)
    return out


@torch.no_grad()
def run_lena(m, x):
    layers = lena_layers(m)
    hooks = []
    for L in layers:
        L.log_stats = True
        hooks.append(L.base.register_forward_hook(lambda mod, i, o, L=L: setattr(L, "_out_norm", o.float().norm(dim=-1))))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        emb = m.tower(pixel_values=x).image_embeds.float()
        logit = m.head(F.normalize(emb, dim=-1) * 10.0).float()
    for hk in hooks:
        hk.remove()
    rec = dict(s=[], F=[], p=[], theta=[], corr=[], fix=[], base=[], uhist=[], u=[], v=[])
    for L in layers:
        d = L.last; u = d["u"].float()                                   # (B, N, r)
        rec["uhist"].append(torch.stack([torch.histc(u[b].clamp(-6, 6), 60, -6, 6) for b in range(u.shape[0])]))
        rec["u"].append(u.half()); rec["v"].append(d["v"].float().half()); rec["p"].append(d["p"].float())
        j, lam = hat_index(u, L.lo, L.hi, L.K)
        th = d["theta"].float()                                           # (B, r, K)
        c = spline_eval(th.unsqueeze(1), j, lam); c0 = spline_eval(L.rho.float(), j, lam)
        Bw = L.B.float() * L.scaling
        rec["corr"].append(((c - c0) @ Bw.T).norm(dim=-1))               # per-image part of the update, (B, N)
        rec["fix"].append((c0 @ Bw.T).norm(dim=-1))                       # shared-curve part
        rec["base"].append(L._out_norm)                                   # norm of W x per token
        for k in ("s", "F"):
            rec[k].append(d[k].float())
        rec["theta"].append(th)
        L.log_stats = False; L.last = None
    out = {k: torch.stack(v, 1).cpu() for k, v in rec.items()}           # (B, M, ...)
    out["emb"] = emb.cpu()
    return logit.float(), out


@torch.no_grad()
def lora_map(m, x):
    maps = []
    hooks = []
    for L in [mod for mod in m.modules() if isinstance(mod, LoRALinear)]:
        def h(mod, inp, out):
            z = inp[0].float()
            maps.append((F.linear(F.linear(z, mod.A1.float()), mod.B.float()) * mod.scaling).norm(dim=-1).cpu())
        hooks.append(L.register_forward_hook(h))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        m(x)
    for hk in hooks:
        hk.remove()
    return torch.stack(maps, 1)                                           # (B, M, N)


@torch.no_grad()
def acc(m, ds, idx, dev, bs=128):
    ok = 0
    for i in range(0, len(idx), bs):
        b = [ds[k] for k in idx[i:i + bs]]
        x = torch.stack([t[0] for t in b]).to(dev); y = torch.tensor([t[1] for t in b], device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ok += (m(x).argmax(-1) == y).sum().item()
    return ok / len(idx)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lena", required=True); p.add_argument("--lora", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n_stats", type=int, default=400, help="images per hospital for curve statistics")
    p.add_argument("--n_ko", type=int, default=2000, help="images per split for the knock-out")
    p.add_argument("--skip_ko", action="store_true")
    p.add_argument("--name", default="diag_cam.pt")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True); dev = "cuda"
    lena_v7.set_triton(False)
    sel = subset(None, a.n_stats)
    m = build("lena", a.lena, dev); ml = build("lora", a.lora, dev)
    names = [n for n, mod in m.tower.named_modules() if isinstance(mod, LeNALinear)]
    res = dict(names=names, hosp=[], y=[], pred=[], s=[], F=[], theta=[], show=[])
    for h in HOSP:
        ds, idx = sel[h]
        for i in range(0, len(idx), 100):
            b = [ds[k] for k in idx[i:i + 100]]
            x = torch.stack([t[0] for t in b]).to(dev)
            logit, r = run_lena(m, x)
            res["hosp"] += [h] * len(b); res["y"] += [t[1] for t in b]; res["pred"] += logit.argmax(-1).tolist()
            for k in ("s", "F", "p", "theta", "uhist", "emb"):
                res.setdefault(k, []).append(r[k].half() if k == "theta" else r[k])
            res.setdefault("ratio_corr", []).append(r["corr"][..., 1:].mean(-1) / r["base"][..., 1:].mean(-1))
            res.setdefault("ratio_fix", []).append(r["fix"][..., 1:].mean(-1) / r["base"][..., 1:].mean(-1))
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                res.setdefault("emb_lora", []).append(ml.tower(pixel_values=x).image_embeds.float().cpu())
            if i == 0:                                                    # maps for the first 8 images of each hospital
                lm = lora_map(ml, x[:8])
                with lena_v7.lena_override(mode="plain"):
                    _, rp = run_lena(m, x[:8])
                for k in range(8):
                    raw = np.array(Image.open(io.BytesIO(ds.imgs[idx[k]])).convert("RGB"))
                    res["show"].append(dict(hosp=h, y=b[k][1], pred=int(logit[k].argmax()), img=raw,
                                            corr=r["corr"][k].numpy(), fix=r["fix"][k].numpy(), lora=lm[k].numpy(),
                                            u=r["u"][k].numpy(), v=r["v"][k].numpy(), theta=r["theta"][k].numpy(),
                                            theta_plain=rp["theta"][k].numpy(), s=r["s"][k].numpy(),
                                            F=r["F"][k].numpy(), p=r["p"][k].numpy()))
        print("hospital", h, "done", flush=True)
    for k in ("s", "F", "p", "theta", "uhist", "emb", "emb_lora", "ratio_corr", "ratio_fix"):
        res[k] = torch.cat(res[k])
    res["rho"] = torch.stack([mod.rho.detach().float().cpu() for mod in m.modules() if isinstance(mod, LeNALinear)])
    # knock-out: s = 0 (shared curve only) in one block or one module type at a time
    tf = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BILINEAR), T.ToTensor(), T.Normalize(MEAN, STD)])
    ko = {}
    for split, keep in (() if a.skip_ko else (("validation", {0, 3, 4}), ("test", None))):
        ds = ParquetImages([os.path.join(D, f"{split}.pkl")], tf, keep=keep, limit=a.n_ko)
        idx = list(range(len(ds))); tag = "id_val" if split == "validation" else "ood_test"
        layers = [(n, mod) for n, mod in m.tower.named_modules() if isinstance(mod, LeNALinear)]
        groups = {"none": []} | {f"block{b}": [mod for n, mod in layers if f"layers.{b}." in n] for b in range(12)} \
            | {f"type_{t}": [mod for n, mod in layers if n.endswith(t)] for t in TARGETS} | {"all": [mod for _, mod in layers]}
        for g, mods in groups.items():
            for mod in mods:
                mod.mode = "fixed"
            ko.setdefault(g, {})[tag] = acc(m, ds, idx, dev)
            for mod in mods:
                mod.mode = "lena"
            print("knockout", tag, g, ko[g][tag], flush=True)
    res["knockout"] = ko
    torch.save(res, os.path.join(a.out, a.name))
    json.dump(ko, open(os.path.join(a.out, "knockout.json"), "w"), indent=1)
    print("DONE")


if __name__ == "__main__":
    main()
