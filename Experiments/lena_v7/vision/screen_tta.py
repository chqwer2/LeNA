"""S1 screen (eval only, no training): test-time adaptation of the shared LeNA curve vs TENT-LN.

Methods (one online stream per split, model + optimiser reset per split, items shuffled once with RandomState(0),
one pass, batch 64; per batch: one Adam step on the prediction entropy, THEN predict the batch with the updated
model under no_grad):
  source   no adaptation
  rho      LeNA only: update the shared curve rho (r x K) and omega of every LeNA layer (vision tower for CLIP); all
           else frozen. The per-input Stein step stays on (theta = rho + s Delta, Delta depends on rho).
  tent_ln  TENT: update the affine weight/bias of every LayerNorm (CLIP: vision tower only, the text features are
           precomputed once; MRI: every LayerNorm-type module of SAM).
Entropy: classification = softmax entropy (CLIP logits scale 100); MRI = binary entropy of the 3 sigmoid channels
(WT/TC/ET) averaged over brain voxels (x != background) of the batch. MRI streams are over visits; a visit's brain
slices are cut into batches of <= bs (update, then predict), non-brain slices are predicted afterwards (no update).
lr selection: every lr in --lrs is run on the ID split only; the lr with the highest ID metric (ties -> smaller lr)
is then run on every OOD split. OOD/test items never influence the choice.

  python screen_tta.py --task camelyon --run runs/cam_lena_lr5e-5 --method rho --lrs 1e-3,1e-2 --out runs/screens/s1/cam_lena_rho
"""
import argparse
import copy
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import eval_switch as ES  # noqa: E402
from lena_v7 import inject, lena_layers  # noqa: E402

DEV = "cuda"


def ac():
    return torch.autocast("cuda", dtype=torch.bfloat16)


def freeze(m):
    for q in m.parameters():
        q.requires_grad_(False)


# --------------------------------------------------------------------------------------------- builders
# return dict(model, fwd(x, st) -> logits, ln_root, lena (list), streams [dict(name, ds | visits, group, ...)], id)
def build_camelyon(run, a):
    from camelyon import Classifier, ParquetImages, TARGETS, MEAN, STD
    from torchvision import transforms as T
    i = ES.info_of(run)
    model = Classifier(i.clip); freeze(model)
    kind = {"lena": "lena", "aurora": "aurora"}.get(i.adapter, "lora")
    kw = dict(r=i.rank, alpha=i.alpha * i.rank / i.r)
    if kind == "lena":
        kw.update(K=16, mode=i.mode, causal=False, use_checkpoint=False)
    inject(model.tower, TARGETS, adapter=kind, **kw)
    ES.load_sd(model, run)
    tf = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BILINEAR), T.ToTensor(), T.Normalize(MEAN, STD)])
    acc = {}
    for split, hs in (("validation", (0, 3, 4, 1)), ("test", (2,))):      # same sampling as eval_switch.py
        ds = ParquetImages([os.path.join(i.data, f"{split}.pkl")], tf); rng = np.random.RandomState(0)
        for h in hs:
            idx = [k for k, c in enumerate(ds.cs) if c == h]
            idx = list(rng.choice(idx, min(a.n, len(idx)), replace=False))
            name = "ID" if h in (0, 3, 4) else f"H{h}"
            d = acc.setdefault(name, dict(name=name, base=ds, idx=[], group=[]))
            d["idx"] += idx; d["group"] += [h] * len(idx)
    streams = [dict(name=d["name"], ds=Subset(d["base"], d["idx"]), group=np.array(d["group"])) for d in acc.values()]
    return dict(model=model, fwd=lambda x, st: model(x), ln_root=model.tower, lena=lena_layers(model),
                streams=streams, id="ID")


def build_clip(run, a):
    from clip_fewshot import CLIPClassifier, Images, classnames, MEAN, STD, PKL
    from torchvision import transforms as T
    i = ES.info_of(run)
    model = CLIPClassifier(i.clip, classnames()); freeze(model)
    kw = dict(r=i.r, alpha=i.alpha * math.sqrt(i.r), dropout=i.dropout)
    if i.adapter == "lena":
        kw.update(K=16, causal=False, use_checkpoint=False, mode=getattr(i, "mode", None) or "lena")
    assert i.adapter in ("lena", "lora"), i.adapter
    inject(model.clip, ["q_proj", "k_proj", "v_proj"], adapter=i.adapter, **kw)
    ES.load_sd(model, run)
    tf = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor(),
                    T.Normalize(MEAN, STD)])
    with torch.no_grad(), ac():
        tfe = model.text_features().float()
    streams = []
    for s in a.clip_sets.split(","):
        ds = Images(os.path.join(PKL, f"{s}.pkl"), tf, limit=a.n, seed=0)
        keep = sorted(set(ds.ys)) if s in ("a", "r") else None
        W = tfe[torch.tensor(keep, device=tfe.device)] if keep else tfe
        remap = None
        if keep:
            remap = torch.full((1000,), -1, dtype=torch.long); remap[torch.tensor(keep)] = torch.arange(len(keep))
        streams.append(dict(name="ID" if s == "val" else s, ds=ds, W=W, remap=remap, group=None))
    vis = [L for n, L in model.named_modules() if L in set(lena_layers(model)) and "vision_model" in n]
    fwd = lambda x, st: 100.0 * model.image_features(x).float() @ st["W"].t()   # noqa: E731
    return dict(model=model, fwd=fwd, ln_root=model.clip.vision_model, lena=vis, streams=streams, id="ID")


def build_domainbed(run, a):
    from camelyon import Classifier, MEAN, STD
    from domainbed_clip import load_splits, DomainImages, classnames, build_adapters
    from torchvision import transforms as T
    i = ES.info_of(run)
    model = Classifier(i.clip, n_cls=len(classnames(i.root, i.dataset))); freeze(model.tower)
    build_adapters(model, i)
    ES.load_sd(model, run)
    tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(MEAN, STD)])
    _, va, te = load_splits(i.root, i.dataset, i.target, i.holdout, i.split_seed, i.limit)
    streams = [dict(name="ID", ds=DomainImages(va, tf), group=None),
               dict(name=f"target:{i.target}", ds=DomainImages(te, tf), group=None)]
    return dict(model=model, fwd=lambda x, st: model(x), ln_root=model.tower, lena=lena_layers(model),
                streams=streams, id="ID")


def build_mri(run, a):
    import mri_sam_seg as M
    i = ES.info_of(run)
    M.BG["value"] = getattr(i, "bg", -3.0)
    sm = getattr(i, "stat_mask", "none")
    model = M.SamSeg(i.sam, i.res, stat_mask=sm if sm != "none" else None)
    i.gate_drop = getattr(i, "gate_drop", 0.0)
    M.build_adapters(model, i)
    ES.load_sd(model, run)
    freeze(model)
    tr_all = M.load_index(i.data, i.train_cohorts, i.limit)
    te = M.load_index(i.data, [c for c in i.test_cohorts if c not in i.train_cohorts], i.limit)
    if any(c in i.train_cohorts for c in i.test_cohorts):
        tr_all, te_in = M.split_patients(tr_all, i.test_holdout, i.seed + 1)
        te += [v for v in te_in if v["cohort"] in i.test_cohorts]
    _, va = M.split_patients(tr_all, i.holdout, i.seed)
    lab = lambda v: v["cohort"] + (f":{v['phase']}" if v.get("phase") and v["cohort"] in M.PHASE_COHORTS else "")  # noqa
    streams = [dict(name="ID", visits=va)]
    for g in sorted({lab(v) for v in te}):
        streams.append(dict(name=g, visits=[v for v in te if lab(v) == g]))
    return dict(model=model, fwd=None, ln_root=model, lena=lena_layers(model), streams=streams, id="ID", M=M,
                in_domain_test=[c for c in i.test_cohorts if c in i.train_cohorts])


BUILD = dict(camelyon=build_camelyon, clip=build_clip, domainbed=build_domainbed, mri=build_mri)


# --------------------------------------------------------------------------------------------- adaptation
def select(B, method):
    model = B["model"]
    for q in model.parameters():
        q.requires_grad_(False)
    ps = []
    if method == "rho":
        assert B["lena"], "rho TTA needs LeNA layers"
        for L in B["lena"]:
            ps += [L.rho, L.omega]
    elif method == "tent_ln":
        for m in B["ln_root"].modules():
            if "LayerNorm" in type(m).__name__ and getattr(m, "weight", None) is not None:
                ps += [m.weight] + ([m.bias] if getattr(m, "bias", None) is not None else [])
    for q in ps:
        q.requires_grad_(True)
    return ps


def ent_cls(logits):
    lf = logits.float()
    return -(lf.softmax(-1) * lf.log_softmax(-1)).sum(-1).mean()


def stream_cls(B, st, opt, a):
    ds = st["ds"]
    order = np.random.RandomState(0).permutation(len(ds))
    if a.limit_stream:
        order = order[:a.limit_stream]
    dl = DataLoader(Subset(ds, order.tolist()), a.bs, shuffle=False, num_workers=a.workers, pin_memory=True)
    out, ents = [], []
    for b in dl:
        x, y = b[0].to(DEV, non_blocking=True), b[1]
        if st.get("remap") is not None:
            y = st["remap"][y]
        if opt is not None:
            with ac():
                lg = B["fwd"](x, st)
            loss = ent_cls(lg)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            ents.append(loss.item())
        with torch.no_grad(), ac():
            pred = B["fwd"](x, st).float().argmax(-1).cpu()
        out.append((pred == y).numpy().astype(float))
    g = st["group"][order] if st.get("group") is not None else np.zeros(len(order), int)
    return np.concatenate(out), order, g, float(np.mean(ents)) if ents else float("nan")


def stream_mri(B, st, opt, a):
    M, model = B["M"], B["model"]
    bgv = M.BG["value"]
    vis = st["visits"]
    order = np.random.RandomState(0).permutation(len(vis))
    if a.max_visits:
        order = order[:a.max_visits]
    if a.limit_stream:
        order = order[:a.limit_stream]
    out, ents = [], []
    for k in order:
        v = vis[k]
        img = np.load(os.path.join(v["path"], "img.npy"), mmap_mode="r")
        gt = M.regions(np.load(os.path.join(v["path"], "seg.npy"))).bool()
        Z = img.shape[0] if not a.max_slices else min(img.shape[0], a.max_slices)
        gt = gt[:Z]
        brain = sorted(int(z) for z in v["brain_slices"] if z < Z)
        rest = [z for z in range(Z) if z not in set(brain)]
        pred = torch.zeros(Z, 3, *gt.shape[-2:], dtype=torch.bool)
        for c0 in range(0, len(brain), a.bs):
            zs = brain[c0:c0 + a.bs]
            x = torch.from_numpy(np.array(img[zs], dtype=np.float32)).to(DEV)
            if opt is not None:
                with ac():
                    lg = model(x)
                lf = lg.float()
                h = F.binary_cross_entropy_with_logits(lf, torch.sigmoid(lf), reduction="none")   # = H(sigmoid(l))
                msk = (x != bgv).any(1, keepdim=True).float()
                loss = (h * msk).sum() / (3 * msk.sum()).clamp_min(1)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                ents.append(loss.item())
            with torch.no_grad(), ac():
                pred[zs] = (model(x) > 0).cpu()
        for c0 in range(0, len(rest), a.bs):
            zs = rest[c0:c0 + a.bs]
            x = torch.from_numpy(np.array(img[zs], dtype=np.float32)).to(DEV)
            with torch.no_grad(), ac():
                pred[zs] = (model(x) > 0).cpu()
        out.append(np.mean([M.dice3d(pred[:, r], gt[:, r]) for r in range(3)]))
    return np.array(out), order, np.zeros(len(order), int), float(np.mean(ents)) if ents else float("nan")


def run_split(B, st, method, lr, init, a):
    B["model"].load_state_dict(init)                                   # reset per split
    ps = select(B, method)
    opt = torch.optim.Adam(ps, lr=lr) if ps else None
    torch.cuda.reset_peak_memory_stats(); t0 = time.time()
    fn = stream_mri if "visits" in st else stream_cls
    m, order, g, ent = fn(B, st, opt, a)
    return dict(split=st["name"], lr=lr, metric=m, item=order, group=g, mean_entropy=ent, n=len(m),
                mean=100 * float(m.mean()), sec=time.time() - t0,
                peak_gb=torch.cuda.max_memory_allocated() / 2**30, n_updated=sum(q.numel() for q in ps))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=list(BUILD), required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--method", choices=["source", "rho", "tent_ln"], required=True)
    p.add_argument("--lrs", default="1e-3,1e-2")
    p.add_argument("--n", type=int, default=3000, help="camelyon: patches per hospital; clip: images per set")
    p.add_argument("--clip_sets", default="val,v2,sketch,a,r")
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--max_visits", type=int, default=0, help="MRI: visits per split (first k of the shuffled order)")
    p.add_argument("--max_slices", type=int, default=0, help="smoke only")
    p.add_argument("--limit_stream", type=int, default=0, help="smoke only: first k items of each stream")
    p.add_argument("--splits", default="", help="smoke only: comma list of split names to run")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    torch.manual_seed(0); np.random.seed(0)
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    B = BUILD[a.task](a.run, a)
    B["model"].to(DEV).eval()
    init = copy.deepcopy(B["model"].state_dict())
    streams = B["streams"]
    if a.splits:
        streams = [s for s in streams if s["name"] in a.splits.split(",")]
    sid = [s for s in streams if s["name"] == B["id"]]
    rows, sel = [], None
    if a.method == "source":
        lrs = [0.0]
    else:
        lrs = [float(x) for x in a.lrs.split(",")]
    for lr in lrs:                                                     # lr selection on the ID split only
        for st in sid:
            r = run_split(B, st, a.method, lr, init, a); r["phase"] = "id_select"; rows.append(r)
            print(f"ID lr={lr:g}: {r['mean']:.3f} ent {r['mean_entropy']:.4f} ({r['n']}, {r['sec']:.0f}s, "
                  f"{r['peak_gb']:.1f} GB, upd {r['n_updated']})", flush=True)
    idr = [r for r in rows if r["split"] == B["id"]]
    sel = max(idr, key=lambda r: (r["mean"], -r["lr"]))["lr"] if idr else lrs[0]
    print(f"selected lr {sel:g} on ID", flush=True)
    for st in streams:
        if st["name"] == B["id"]:
            continue
        r = run_split(B, st, a.method, sel, init, a); r["phase"] = "ood"; rows.append(r)
        print(f"{st['name']} lr={sel:g}: {r['mean']:.3f} ent {r['mean_entropy']:.4f} ({r['n']}, {r['sec']:.0f}s, "
              f"{r['peak_gb']:.1f} GB)", flush=True)
    np.savez_compressed(os.path.join(a.out, "items.npz"),
                        **{f"{r['phase']}|{r['split']}|{r['lr']:g}|{k}": r[k] for r in rows
                           for k in ("metric", "item", "group")})
    summ = [{k: v for k, v in r.items() if k not in ("metric", "item", "group")} for r in rows]
    json.dump(dict(args=vars(a), selected_lr=sel, rows=summ, wall_s=time.time() - t0),
              open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print(f"DONE {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
