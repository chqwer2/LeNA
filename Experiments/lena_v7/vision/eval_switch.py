"""Shift-gated switch between a LoRA model and a LeNA model (eval only; no training).

Rule (the Camelyon gated hybrid of AGENDA 2026-10-21 05:50 / results/gate_switch/analyze_gate_switch.py, generalised):
  1. Run the trained LeNA model on an input x and read, for every LeNA layer l and rank channel c, the F statistic
     of its per-input fit (lena_v7.fit_and_step: F = S / (p * sigma^2), the evidence that the input's curve departs
     from the shared prior; 0 when the fit is not identifiable). Feature f_lc(x) = log(1 + F_lc(x)).
     Vision encoders: only the image-encoder LeNA layers. MRI: f is averaged over the brain slices of the volume
     (one score per volume; the switch routes the whole volume).
  2. Calibration set C = a random half (numpy default_rng(seed 0)) of the IN-DOMAIN validation data only
     (Camelyon: ID-hospital patches 0/3/4; CLIP: ImageNet val; DomainBed: the 20 % held-out source images;
     MRI: the patient-level hold-out of the training cohort). Nothing from any OOD / target / test set is used.
     med_lc, IQR_lc = median and inter-quartile range of f_lc over C.
  3. score(x) = mean_lc |f_lc(x) - med_lc| / (IQR_lc + 1e-6);  tau = q-quantile of score over C (q = 0.8 default).
  4. Prediction = LeNA's if score(x) > tau, else LoRA's.
  ID numbers are reported on the other half of the ID validation data (never on C); OOD sets on all their items.

Output: <out>/switch.csv (per split: n, LoRA, LeNA, switch, % routed to LeNA, paired-bootstrap 95% CI of
switch-LoRA and switch-LeNA in percentage points) and <out>/switch_items.npz (per item: split, metric of each model,
score, routed). Metric = accuracy (classification) or mean Dice over WT/TC/ET (MRI, per volume).

  python eval_switch.py --task camelyon --lena runs/cam_lena_lr5e-5 --lora runs/cam_lora_lr1e-4 --out runs/switch_cam
  python eval_switch.py --task clip --lena runs/clip_lena_s1 --lora runs/clip_lora_s1 --out runs/switch_clip
  python eval_switch.py --task domainbed --lena runs/db_oh_art_lena_s0 --lora runs/db_oh_art_lora_s0 --out runs/switch_oh_art
  python eval_switch.py --task mri --lena runs/mri_diag_lena_lr2e-4 --lora runs/mri_sw_lora_lr2e-4 --out runs/switch_mri
"""
import argparse
import contextlib
import csv
import json
import math
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
from lena_v7 import inject, lena_layers  # noqa: E402

DEV = {"dev": "cuda", "dtype": torch.bfloat16}


def autocast():
    d = DEV["dev"]
    return torch.autocast(d if d != "cpu" else "cpu", dtype=DEV["dtype"])


def info_of(run):
    return SimpleNamespace(**json.load(open(os.path.join(run, "result.json")))["info"])


def ckpt_of(run):
    for f in ("best.pt", "adapter.pt"):
        if os.path.exists(os.path.join(run, f)):
            return os.path.join(run, f)
    raise FileNotFoundError(run)


def load_sd(model, run):
    sd = torch.load(ckpt_of(run), map_location="cpu")
    bad = [k for k in sd if k not in model.state_dict()]
    assert not bad, f"{run}: checkpoint keys do not match the model: {bad[:3]}"
    model.load_state_dict(sd, strict=False)
    return model.to(DEV["dev"]).eval()


class FLog:
    """Context: log the F statistic of the given LeNA layers for one forward; .get(b) -> (b, M, r) float cpu."""

    def __init__(self, layers):
        self.layers = layers

    def __enter__(self):
        for L in self.layers:
            L.log_stats = True
        return self

    def get(self, b):
        out = []
        for L in self.layers:
            t = L.last["F"].float()
            t = t.reshape(b, -1, t.shape[-1]).mean(1) if t.shape[0] != b else t      # several windows per image (batch-major)
            out.append(t.cpu())
        return torch.stack(out, 1)

    def __exit__(self, *a):
        for L in self.layers:
            L.log_stats = False; L.last = None


# ------------------------------------------------------------------------------------------------- tasks
# Each task returns {split: dict(metric=np.array, feat=np.array or None)} for one model, with the same item order for
# both models, plus the name of the ID split.

def task_camelyon(run, is_lena, a):
    from camelyon import Classifier, ParquetImages, TARGETS, MEAN, STD
    from torchvision import transforms as T
    i = info_of(run)
    model = Classifier(i.clip)
    for q in model.tower.parameters():
        q.requires_grad_(False)
    kind = {"lena": "lena", "aurora": "aurora"}.get(i.adapter, "lora")
    kw = dict(r=i.rank, alpha=i.alpha * i.rank / i.r)
    if i.adapter == "dora":
        kw["dora"] = True
    if kind == "lena":
        kw.update(K=16, mode=i.mode, causal=False, use_checkpoint=False)
    inject(model.tower, TARGETS, adapter=kind, **kw)
    load_sd(model, run)
    tf = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BILINEAR), T.ToTensor(), T.Normalize(MEAN, STD)])
    layers = lena_layers(model)
    out = {}
    for split, hs in (("validation", (0, 3, 4, 1)), ("test", (2,))):    # same sampling as diag_gate_switch.py
        ds = ParquetImages([os.path.join(i.data, f"{split}.pkl")], tf); rng = np.random.RandomState(0)
        for h in hs:
            idx = [k for k, c in enumerate(ds.cs) if c == h]
            idx = list(rng.choice(idx, min(a.n, len(idx)), replace=False))
            m, f = [], []
            for x, y, _ in DataLoader(Subset(ds, idx), a.bs, num_workers=a.workers):
                x = x.to(DEV["dev"])
                with torch.no_grad(), (FLog(layers) if is_lena else contextlib.nullcontext()) as lg, autocast():
                    pred = model(x).float().argmax(-1).cpu()
                    if is_lena:
                        f.append(lg.get(len(y)))
                m.append((pred == y).numpy().astype(float))
            name = "ID" if h in (0, 3, 4) else f"H{h}"
            prev = out.get(name, dict(metric=[], feat=[], group=[]))
            prev["metric"].append(np.concatenate(m)); prev["group"].append(np.full(len(idx), h))
            if is_lena:
                prev["feat"].append(torch.cat(f).numpy())
            out[name] = prev
            print(f"  {os.path.basename(run)} H{h}: {np.concatenate(m).mean():.4f} ({len(idx)})", flush=True)
    return {k: dict(metric=np.concatenate(v["metric"]), group=np.concatenate(v["group"]),
                    feat=np.concatenate(v["feat"]) if v["feat"] else None) for k, v in out.items()}, "ID"


def task_clip(run, is_lena, a):
    from clip_fewshot import CLIPClassifier, Images, classnames, MEAN, STD, PKL
    from torchvision import transforms as T
    i = info_of(run)
    model = CLIPClassifier(i.clip, classnames())
    for q in model.parameters():
        q.requires_grad_(False)
    kw = dict(r=i.r, alpha=i.alpha * math.sqrt(i.r), dropout=i.dropout)
    if i.adapter == "lena":
        kw.update(K=16, causal=False, use_checkpoint=False, mode=getattr(i, "mode", None) or "lena")
    if i.adapter == "dora":
        kw["dora"] = True
    assert i.adapter in ("lena", "lora", "dora", "aurora"), i.adapter
    inject(model.clip, ["q_proj", "k_proj", "v_proj"], adapter={"dora": "lora"}.get(i.adapter, i.adapter), **kw)
    load_sd(model, run)
    tf_te = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor(),
                       T.Normalize(MEAN, STD)])
    vis = [L for n, L in model.named_modules() if L in set(lena_layers(model)) and "vision_model" in n]
    with torch.no_grad(), autocast():
        tfe = model.text_features().float()
    out = {}
    for s in a.clip_sets.split(","):
        ds = Images(os.path.join(PKL, f"{s}.pkl"), tf_te, limit=a.n, seed=0)
        keep = sorted(set(ds.ys)) if s in ("a", "r") else None
        W = tfe[torch.tensor(keep, device=tfe.device)] if keep else tfe
        remap = None
        if keep:
            remap = torch.full((1000,), -1, dtype=torch.long); remap[torch.tensor(keep)] = torch.arange(len(keep))
        m, f = [], []
        for x, y in DataLoader(ds, a.bs, num_workers=a.workers):
            x = x.to(DEV["dev"])
            with torch.no_grad(), (FLog(vis) if is_lena else contextlib.nullcontext()) as lg, autocast():
                pred = (model.image_features(x).float() @ W.t()).argmax(-1).cpu()
                if is_lena:
                    f.append(lg.get(len(y)))
            yy = remap[y] if keep else y
            m.append((pred == yy).numpy().astype(float))
        out["ID" if s == "val" else s] = dict(metric=np.concatenate(m), group=None,
                                              feat=torch.cat(f).numpy() if is_lena else None)
        print(f"  {os.path.basename(run)} {s}: {out['ID' if s == 'val' else s]['metric'].mean():.4f}", flush=True)
    return out, "ID"


def task_domainbed(run, is_lena, a):
    from camelyon import Classifier, MEAN, STD
    from domainbed_clip import load_splits, DomainImages, classnames, build_adapters
    from torchvision import transforms as T
    i = info_of(run)
    names = classnames(i.root, i.dataset)
    model = Classifier(i.clip, n_cls=len(names))
    for q in model.tower.parameters():
        q.requires_grad_(False)
    build_adapters(model, i)
    load_sd(model, run)
    tf_te = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(MEAN, STD)])
    _, va, te = load_splits(i.root, i.dataset, i.target, i.holdout, i.split_seed, a.limit or i.limit)
    layers = lena_layers(model)
    out = {}
    for name, items in (("ID", va), (f"target:{i.target}", te)):
        m, f, g = [], [], []
        for x, y, d in DataLoader(DomainImages(items, tf_te), a.bs, num_workers=a.workers):
            x = x.to(DEV["dev"])
            with torch.no_grad(), (FLog(layers) if is_lena else contextlib.nullcontext()) as lg, autocast():
                pred = model(x).float().argmax(-1).cpu()
                if is_lena:
                    f.append(lg.get(len(y)))
            m.append((pred == y).numpy().astype(float)); g.append(d.numpy())
        out[name] = dict(metric=np.concatenate(m), group=np.concatenate(g), feat=torch.cat(f).numpy() if is_lena else None)
        print(f"  {os.path.basename(run)} {name}: {out[name]['metric'].mean():.4f} ({len(items)})", flush=True)
    return out, "ID"


def task_mri(run, is_lena, a):
    import mri_sam_seg as M
    i = info_of(run)
    for flag in ("ridge", "pool_scan", "smooth", "aux_shift"):
        assert not getattr(i, flag, False), f"{run}: --{flag} runs are not supported by the switch eval"
    M.BG["value"] = getattr(i, "bg", -3.0)
    sm = getattr(i, "stat_mask", "none")
    model = M.SamSeg(i.sam, i.res, stat_mask=sm if sm != "none" else None)
    i.gate_drop = getattr(i, "gate_drop", 0.0)
    M.build_adapters(model, i)
    load_sd(model, run)
    tr_all = M.load_index(i.data, i.train_cohorts, a.limit or i.limit)
    if a.mri_test:
        i.test_cohorts = a.mri_test.split(",")
    te = M.load_index(i.data, [c for c in i.test_cohorts if c not in i.train_cohorts], a.limit or i.limit)
    if any(c in i.train_cohorts for c in i.test_cohorts):
        tr_all, te_in = M.split_patients(tr_all, i.test_holdout, i.seed + 1)
        te += [v for v in te_in if v["cohort"] in i.test_cohorts]
    _, va = M.split_patients(tr_all, i.holdout, i.seed)
    layers = lena_layers(model)
    out = {}
    for name, visits in (("ID", va), ("test", te)):
        m, f, coh = [], [], []
        for v in visits:
            img = np.load(os.path.join(v["path"], "img.npy"), mmap_mode="r")
            gt = M.regions(np.load(os.path.join(v["path"], "seg.npy"))).bool()
            preds, fs = [], []
            Z = min(img.shape[0], a.max_slices) if a.max_slices else img.shape[0]
            gt = gt[:Z]
            for z0 in range(0, Z, a.bs):
                x = torch.from_numpy(np.array(img[z0:min(Z, z0 + a.bs)], dtype=np.float32)).to(DEV["dev"])
                with torch.no_grad(), (FLog(layers) if is_lena else contextlib.nullcontext()) as lg, autocast():
                    preds.append((model(x) > 0).cpu())
                    if is_lena:
                        fs.append(lg.get(x.shape[0]))
            pred = torch.cat(preds)
            m.append(np.mean([M.dice3d(pred[:, k], gt[:, k]) for k in range(3)]))
            if is_lena:
                fz = torch.cat(fs)                                       # (Z, M, r)
                zs = np.array([z for z in v["brain_slices"] if z < Z]) if any(z < Z for z in v["brain_slices"]) else np.arange(fz.shape[0])
                f.append(torch.log1p(fz[zs]).mean(0).numpy())           # volume feature = mean of log(1+F) over brain slices
            coh.append(v["cohort"] + (f":{v['phase']}" if v.get("phase") and v["cohort"] in M.PHASE_COHORTS else ""))
        out[name] = dict(metric=np.array(m), group=np.array(coh), feat=np.stack(f) if is_lena else None, logged=True)
        print(f"  {os.path.basename(run)} {name}: mean Dice {np.mean(m):.4f} ({len(visits)} visits)", flush=True)
    return out, "ID"


TASKS = dict(camelyon=task_camelyon, clip=task_clip, domainbed=task_domainbed, mri=task_mri)


# ------------------------------------------------------------------------------------------------- switch
def features(d):
    f = d["feat"].reshape(len(d["feat"]), -1).astype(np.float64)
    return f if d.get("logged") else np.log1p(np.clip(f, 0, None))


def boot(a_, b_, groups=None, B=1000, seed=0):
    """Paired bootstrap of mean(a - b) in percentage points (stratified by group if given; macro over groups)."""
    d = 100 * (a_ - b_); rng = np.random.default_rng(seed)
    gs = [np.arange(len(d))] if groups is None else [np.where(groups == g)[0] for g in np.unique(groups)]
    pt = np.mean([d[g].mean() for g in gs])
    est = [np.mean([d[g[rng.integers(0, len(g), len(g))]].mean() for g in gs]) for _ in range(B)]
    return pt, np.percentile(est, 2.5), np.percentile(est, 97.5)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=list(TASKS), required=True)
    p.add_argument("--lena", required=True, help="LeNA run dir (result.json + best.pt/adapter.pt)")
    p.add_argument("--lora", required=True, help="LoRA run dir")
    p.add_argument("--q", type=float, default=0.8, help="threshold quantile of the score on the ID calibration half")
    p.add_argument("--n", type=int, default=3000, help="camelyon: patches per hospital; clip: images per set (0 = all)")
    p.add_argument("--clip_sets", default="val,v2,sketch,a,r")
    p.add_argument("--limit", type=int, default=0, help="smoke: domainbed images per domain / mri visits per cohort")
    p.add_argument("--max_slices", type=int, default=0, help="smoke only: first k slices of each MRI volume")
    p.add_argument("--mri_test", default="", help="smoke only: comma list restricting the MRI test cohorts")
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    DEV["dev"] = a.device
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    fn = TASKS[a.task]
    lena, idn = fn(a.lena, True, a)
    lora, _ = fn(a.lora, False, a)
    for k in lena:
        assert len(lena[k]["metric"]) == len(lora[k]["metric"]), k
    # calibration: random half of the ID split, chosen without looking at any label or OOD item
    nid = len(lena[idn]["metric"]); rng = np.random.default_rng(0)
    assert nid >= 4, f"need >= 4 ID validation items for a calibration/eval split, got {nid} (smoke: raise --limit)"
    cal = np.zeros(nid, bool); cal[rng.choice(nid, nid // 2, replace=False)] = True
    fid = features(lena[idn])
    med = np.median(fid[cal], 0); iqr = np.subtract(*np.percentile(fid[cal], [75, 25], 0)) + 1e-6
    score = lambda f: np.abs((f - med) / iqr).mean(1)                    # noqa: E731
    tau = np.percentile(score(fid[cal]), 100 * a.q)
    rows, items = [], dict(split=[], lena=[], lora=[], switch=[], score=[], routed=[], group=[])
    for k in lena:
        sel = ~cal if k == idn else np.ones(len(lena[k]["metric"]), bool)
        sc = score(features(lena[k]))[sel]; r = sc > tau
        ml, mo = lena[k]["metric"][sel], lora[k]["metric"][sel]
        sw = np.where(r, ml, mo)
        g = lena[k]["group"][sel] if lena[k].get("group") is not None else None
        row = dict(split=k + (" (eval half)" if k == idn else ""), n=int(sel.sum()), lora=100 * mo.mean(),
                   lena=100 * ml.mean(), switch=100 * sw.mean(), routed_pct=100 * r.mean())
        for nm, ref in (("lora", mo), ("lena", ml)):
            pt, lo, hi = boot(sw, ref)
            row.update({f"switch_minus_{nm}": pt, f"switch_minus_{nm}_lo": lo, f"switch_minus_{nm}_hi": hi})
        rows.append(row)
        for key, val in (("split", np.full(sel.sum(), k)), ("lena", ml), ("lora", mo), ("switch", sw), ("score", sc),
                         ("routed", r), ("group", g if g is not None else np.full(sel.sum(), ""))):
            items[key].append(np.asarray(val).astype(str) if key in ("split", "group") else val)
        if g is not None and len(np.unique(g)) > 1:                       # per-group rows (hospital / cohort / domain)
            for gg in np.unique(g):
                s_ = g == gg
                rows.append(dict(split=f"{k}/{gg}", n=int(s_.sum()), lora=100 * mo[s_].mean(), lena=100 * ml[s_].mean(),
                                 switch=100 * sw[s_].mean(), routed_pct=100 * r[s_].mean()))
    rows.append(dict(split=f"threshold tau (q={a.q}, ID calibration half n={int(cal.sum())})", n=int(cal.sum()),
                     lora=float(tau)))
    keys = list(dict.fromkeys(k for r_ in rows for k in r_))
    with open(os.path.join(a.out, "switch.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, keys); w.writeheader()
        w.writerows([{k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r_.items()} for r_ in rows])
    np.savez_compressed(os.path.join(a.out, "switch_items.npz"), **{k: np.concatenate(v) for k, v in items.items()},
                        tau=tau, med=med, iqr=iqr, q=a.q)
    json.dump(dict(vars(a), tau=float(tau), wall_s=time.time() - t0), open(os.path.join(a.out, "switch_args.json"), "w"))
    for r_ in rows:
        print(" | ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in r_.items()), flush=True)
    print(f"DONE {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
