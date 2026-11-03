"""Usability study, inference only (no training): per-input outputs of finished LoRA / DoRA / AuroRA / LeNA / LeNA-S
checkpoints for shift detection, selective prediction, dose-response, calibration and the per-layer atlas.

New file; imports camelyon.py / clip_fewshot.py / mri_sam_seg.py / domainbed_clip.py unchanged.

Per input (classification): y, pred, msp, energy (logsumexp of the logits), nll (-log p_y), max-logit.
Per input, LeNA only: log1p(F) per layer x channel (fp16), mean Stein step s per layer, and per layer the token
variance of the applied correction delta(u) = phi_theta(u) - phi_rho(u) and the part explained by an affine map of
the prior curve, a*phi_rho(u) + b (least squares over the input's tokens) -> 'nonlinearity used' = 1 - aff/var.
MRI: the same per visit (aggregated over the visit's brain slices) plus Dice and a voxel-entropy baseline.

  python usab_infer.py --task cam --adapter lena --ckpt runs/cam_lena_lr5e-5/best.pt --out runs/usab_v1/cam_lena.npz
"""
import argparse
import json
import math
import os
import sys
import time
import types

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import lena_v7 as LV  # noqa: E402
from lena_v7 import inject, lena_layers, hat_index  # noqa: E402


class LeNARec:
    """Forward hooks on LeNA layers: per-sample statistics, then frees L.last."""

    def __init__(self, layers):
        self.layers = layers
        self.buf = None
        for i, L in enumerate(layers):
            L.log_stats = True
            L.register_forward_hook(lambda mod, inp, out, i=i: self._hook(i, mod))

    def start(self):
        self.buf = {}

    def _hook(self, i, L):
        d = L.last
        if d is None or self.buf is None:
            return
        Fk = d["F"].float()                                           # (B, r)
        s = d["s"].float()
        u = d["u"].float()                                            # (B, N, r)
        th = d["theta"].float()                                       # (B, r, K)
        rho = L.rho.float()
        j, lam = hat_index(u, L.lo, L.hi, L.K)
        dl = LV.spline_eval((th - rho).unsqueeze(1), j, lam)          # applied correction at the tokens (B, N, r)
        x = LV.spline_eval(rho, j, lam)                               # prior curve at the tokens
        mx, my = x.mean(1), dl.mean(1)
        vx = (x * x).mean(1) - mx ** 2
        vy = ((dl * dl).mean(1) - my ** 2).clamp_min(0)
        cxy = (x * dl).mean(1) - mx * my
        aff = torch.where(vx > 1e-10, cxy ** 2 / vx.clamp_min(1e-10), torch.zeros_like(vx)).clamp(max=vy)
        self.buf[i] = (torch.log1p(Fk.clamp_min(0)), s.mean(-1), vy.sum(-1), aff.sum(-1))
        L.last = None

    def collect(self):
        M = len(self.layers)
        lf = torch.stack([self.buf[i][0] for i in range(M)], 1)       # (B, M, r)
        out = dict(logF=lf.half().cpu(), s=torch.stack([self.buf[i][1] for i in range(M)], 1).half().cpu(),
                   nl_var=torch.stack([self.buf[i][2] for i in range(M)], 1).cpu(),
                   nl_aff=torch.stack([self.buf[i][3] for i in range(M)], 1).cpu())
        self.buf = None
        return out


def cls_stats(logits, y):
    lp = logits.float().log_softmax(-1)
    return dict(y=y.cpu(), pred=lp.argmax(-1).cpu(), msp=lp.max(-1).values.exp().cpu(),
                energy=torch.logsumexp(logits.float(), -1).cpu(), maxlogit=logits.float().max(-1).values.cpu(),
                nll=(-lp.gather(-1, y[:, None]).squeeze(-1)).cpu())


def install_neff(model, neff):
    if neff:
        import lena_mri_ext
        lena_mri_ext.install(model, neff=True)
        lena_mri_ext.EXT["deff_scale"] = 1.0


def load(model, path, dev):
    sd = torch.load(path, map_location="cpu")
    miss = [k for k in sd if k not in model.state_dict()]
    assert not miss, f"checkpoint keys do not match the model: {miss[:3]}"
    model.load_state_dict({k: v.to(dev) for k, v in sd.items()}, strict=False)


def run_loader(model_fn, loader, rec, extra=None):
    out = {}
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0].cuda(non_blocking=True), batch[1].cuda()
            if rec is not None:
                rec.start()
            logits = model_fn(x)
            r = cls_stats(logits, y)
            if rec is not None:
                r.update(rec.collect())
            if extra is not None:
                r["group"] = torch.as_tensor(batch[2])
            for k, v in r.items():
                out.setdefault(k, []).append(v)
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def save(path, sets, meta):
    flat = {}
    for s, d in sets.items():
        for k, v in d.items():
            flat[f"{s}/{k}"] = v
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez(path, meta=json.dumps(meta), **flat)
    print("SAVED", path, {s: len(d["y"]) if "y" in d else len(d.get("mean", [])) for s, d in sets.items()}, flush=True)


# ---------------------------------------------------------------------------------------------------------------
def task_cam(a, dev):
    import camelyon as C
    from torchvision import transforms as T
    tf = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BILINEAR), T.ToTensor(), T.Normalize(C.MEAN, C.STD)])
    g = lambda s: [os.path.join(C_DATA, f"{s}.pkl")]  # noqa: E731
    ev = {"id_val": C.ParquetImages(g("validation"), tf, keep={0, 3, 4}, limit=a.limit),
          "ood_val": C.ParquetImages(g("validation"), tf, keep={1}, limit=a.limit),
          "ood_test": C.ParquetImages(g("test"), tf, limit=a.limit)}
    model = C.Classifier("openai/clip-vit-base-patch16")
    for q in model.tower.parameters():
        q.requires_grad_(False)
    kind = {"lena": "lena", "aurora": "aurora"}.get(a.adapter, "lora")
    kw = dict(r=8, alpha=16)
    if a.adapter == "dora":
        kw["dora"] = True
    if kind == "lena":
        kw.update(K=16, mode="lena", causal=False, use_checkpoint=False)
    inject(model.tower, C.TARGETS, adapter=kind, **kw)
    install_neff(model, a.neff)
    model.to(dev)
    load(model, a.ckpt, dev)
    model.eval()
    layers = [L for n, L in model.named_modules() if isinstance(L, LV.LeNALinear) and "encoder.layers" in n]
    rec = LeNARec(layers) if layers else None

    def fn(x):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return model(x).float()
    sets = {}
    for k, ds in ev.items():
        t0 = time.time()
        sets[k] = run_loader(fn, DataLoader(ds, a.bs, num_workers=a.workers, pin_memory=True), rec, extra=True)
        print(k, len(ds), "acc %.4f" % (sets[k]["pred"] == sets[k]["y"]).mean(), f"{time.time() - t0:.0f}s", flush=True)
    return sets, dict(layers=[n for n, L in model.named_modules() if isinstance(L, LV.LeNALinear) and "encoder.layers" in n])


C_DATA = "/mnt/scratch2/users/hchen/data/lena_v7/camelyon17/pkl"


def task_clip(a, dev):
    import clip_fewshot as CF
    from torchvision import transforms as T
    tf = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor(),
                    T.Normalize(CF.MEAN, CF.STD)])
    model = CF.CLIPClassifier("openai/clip-vit-base-patch16", CF.classnames())
    for q in model.parameters():
        q.requires_grad_(False)
    kw = dict(r=2, alpha=1.0 * math.sqrt(2), dropout=0.25)
    kind = {"dora": "lora"}.get(a.adapter, a.adapter)
    if a.adapter == "dora":
        kw["dora"] = True
    if kind == "lena":
        kw.update(K=16, causal=False, use_checkpoint=False, mode="lena")
    inject(model.clip, ["q_proj", "k_proj", "v_proj"], adapter=kind, **kw)
    install_neff(model, a.neff)
    model.cuda()
    load(model, a.ckpt, dev)
    model.eval()
    vis_names = [n for n, L in model.named_modules() if isinstance(L, LV.LeNALinear) and "vision_model" in n]
    vis = [model.get_submodule(n) for n in vis_names]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        tfeat = model.text_features().float()                         # text LeNA layers: log_stats off
    rec = LeNARec(vis) if vis else None
    sets = {}
    for s in a.sets.split(","):
        ds = CF.Images(os.path.join(CF.PKL, f"{s}.pkl"), tf, limit=a.limit)
        t = tfeat
        if s in ("a", "r") and not a.full_head:
            keep = sorted(set(ds.ys))
            remap = {c: i for i, c in enumerate(keep)}
            ds.ys = [remap[c] for c in ds.ys]
            t = tfeat[torch.tensor(keep, device=tfeat.device)]

        def fn(x, t=t):
            with torch.autocast("cuda", dtype=torch.float16):
                return 100.0 * model.image_features(x).float() @ t.t()
        t0 = time.time()
        sets[s] = run_loader(fn, DataLoader(ds, a.bs, num_workers=a.workers, pin_memory=True), rec)
        print(s, len(ds), "acc %.4f" % (sets[s]["pred"] == sets[s]["y"]).mean(), f"{time.time() - t0:.0f}s", flush=True)
    return sets, dict(layers=vis_names)


def task_oh(a, dev):
    import domainbed_clip as DB
    import camelyon as C
    from torchvision import transforms as T
    tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(C.MEAN, C.STD)])
    tr, va, te = DB.load_splits(DB.ROOT, "OfficeHome", "Art", 0.2, 0)
    names = DB.classnames(DB.ROOT, "OfficeHome")
    model = C.Classifier("openai/clip-vit-base-patch16", n_cls=len(names))
    DB.zeroshot_head(model, "openai/clip-vit-base-patch16", names)
    for q in model.tower.parameters():
        q.requires_grad_(False)
    DB.build_adapters(model, types.SimpleNamespace(adapter=a.adapter, r=8, alpha=16, mode="lena"))
    install_neff(model, a.neff)
    model.to(dev)
    load(model, a.ckpt, dev)
    model.eval()
    names_l = [n for n, L in model.named_modules() if isinstance(L, LV.LeNALinear) and "encoder.layers" in n]
    rec = LeNARec([model.get_submodule(n) for n in names_l]) if names_l else None

    def fn(x):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return model(x).float()
    sets = {}
    for k, v in (("id_val", va), ("target", te)):
        sets[k] = run_loader(fn, DataLoader(DB.DomainImages(v, tf), a.bs, num_workers=a.workers, pin_memory=True), rec, extra=True)
        print(k, len(v), "acc %.4f" % (sets[k]["pred"] == sets[k]["y"]).mean(), flush=True)
    return sets, dict(layers=names_l)


def task_mri(a, dev):
    import mri_sam_seg as M
    model = M.SamSeg(M.SAM, 224)
    M.build_adapters(model, types.SimpleNamespace(adapter=a.adapter, r=8, alpha=16, mode="lena", targets=["qkv"],
                                                  gate_drop=0.0, knots=16))
    install_neff(model, a.neff)
    model.to(dev)
    load(model, a.ckpt, dev)
    model.eval()
    names_l = [n for n, L in model.named_modules() if isinstance(L, LV.LeNALinear)]
    rec = LeNARec([model.get_submodule(n) for n in names_l]) if names_l else None
    gli = M.load_index(M.DATA, ["BraTS-2023-GLI"], a.limit)
    cohorts = ["BraTS-2023-SSA", "BraTS-2023-PED", "RHUH-GBM", "IVYGAP2", "SAILOR", "UCSD-PTGBM"]
    te = M.load_index(M.DATA, cohorts, a.limit)
    tr_all, te_in = M.split_patients(gli, 0.2, 1)                     # same splits as the runs (seed 0)
    tr, va = M.split_patients(tr_all, 0.1, 0)
    rng = np.random.RandomState(0)
    fit = [tr[i] for i in rng.choice(len(tr), min(a.mri_fit, len(tr)), replace=False)]
    groups = {"fit": fit, "calib": va, "test": te_in + te}
    sets = {}
    for gname, visits in groups.items():
        rows = {}
        t0 = time.time()
        for v in visits:
            img = np.load(os.path.join(v["path"], "img.npy"), mmap_mode="r")
            gt = M.regions(np.load(os.path.join(v["path"], "seg.npy"))).bool()
            bsl = set(int(z) for z in v.get("brain_slices", range(img.shape[0])))
            preds, ent_num, ent_den, conf_num, conf_den = [], 0.0, 0.0, 0.0, 0.0
            acc = dict(logF=[], s=[], nl_var=[], nl_aff=[])
            with torch.no_grad():
                for z0 in range(0, img.shape[0], 40):
                    x = torch.from_numpy(np.array(img[z0:z0 + 40], dtype=np.float32)).to(dev)
                    if rec is not None:
                        rec.start()
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        lo = model(x).float()
                    pr = torch.sigmoid(lo)
                    preds.append((lo > 0).cpu())
                    brain = (x != M.BG["value"]).any(1, keepdim=True).expand_as(pr)
                    H = -(pr * pr.clamp_min(1e-6).log() + (1 - pr) * (1 - pr).clamp_min(1e-6).log())
                    ent_num += float(H[brain].sum()); ent_den += float(brain.sum())
                    fg = lo[:, 0] > 0
                    conf_num += float(pr[:, 0][fg].sum()); conf_den += float(fg.sum())
                    if rec is not None:
                        st = rec.collect()
                        keep = torch.tensor([(z0 + i) in bsl for i in range(x.shape[0])])
                        if keep.any():
                            for k in acc:
                                acc[k].append(st[k][keep].float())
            pred = torch.cat(preds)
            r = dict(cohort=v["cohort"], phase=v.get("phase", ""), patient=str(v["patient"]), visit=str(v["visit"]),
                     **{rg: M.dice3d(pred[:, k], gt[:, k]) for k, rg in enumerate(M.REGIONS)},
                     entropy=ent_num / max(ent_den, 1), conf_fg=conf_num / conf_den if conf_den else 1.0)
            r["mean"] = float(np.mean([r[rg] for rg in M.REGIONS]))
            if rec is not None and acc["logF"]:
                r["logF"] = torch.cat(acc["logF"]).mean(0).numpy()            # (M, r)
                r["s"] = torch.cat(acc["s"]).mean(0).numpy()
                r["nl_var"] = torch.cat(acc["nl_var"]).sum(0).numpy()
                r["nl_aff"] = torch.cat(acc["nl_aff"]).sum(0).numpy()
            for k, val in r.items():
                rows.setdefault(k, []).append(val)
        if not rows:
            print(gname, "EMPTY", flush=True)
            continue
        sets[gname] = {k: np.array(v) for k, v in rows.items()}
        print(gname, len(visits), "dice %.4f" % np.mean(sets[gname]["mean"]), f"{time.time() - t0:.0f}s", flush=True)
    return sets, dict(layers=names_l)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["cam", "clip", "oh", "mri"], required=True)
    p.add_argument("--adapter", choices=["lena", "lora", "dora", "aurora"], required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--neff", action="store_true")
    p.add_argument("--sets", default="val50k,v2,sketch,a,r")
    p.add_argument("--full_head", action="store_true", help="CLIP a/r: logits over all 1000 classes (OOD-score baselines)")
    p.add_argument("--limit", type=int, default=0, help="smoke: items per split / visits per cohort")
    p.add_argument("--mri_fit", type=int, default=100, help="MRI: training visits used to fit the score scale")
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    torch.manual_seed(0)
    dev = "cuda"
    t0 = time.time()
    sets, meta = {"cam": task_cam, "clip": task_clip, "oh": task_oh, "mri": task_mri}[a.task](a, dev)
    meta.update(vars(a), seconds=time.time() - t0)
    save(a.out, sets, meta)
    print("DONE", f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
