"""R3 diagnostic (eval only, no training): quantile-matching fit on an existing DomainBed LeNA checkpoint.
1. standard LeNA: accuracy + per-image F features on ID val and the target domain;
2. calibration pass on source training images -> per layer/channel reference u-quantiles;
3. R3 fit (theta from quantile matching, F = quantile distance): accuracy + features.
For both feature sets: eval_switch's routing score (median/IQR on a seed-0 half of ID val, tau = q-quantile),
% routed on target vs ID eval half, and AUROC(score: target vs ID eval half).

  python r3_diag.py --run runs/db_oh_art_lena_s0 --n_calib 2000 --out runs/r3diag_oh_art
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_switch as ES  # noqa: E402
import lena_mri_ext as X  # noqa: E402
from lena_v7 import lena_layers  # noqa: E402


def run_split(model, layers, ds, bs, workers):
    acc, feats = [], []
    for x, y, _ in DataLoader(ds, bs, num_workers=workers):
        x = x.cuda()
        with torch.no_grad(), ES.FLog(layers) as lg, ES.autocast():
            pred = model(x).float().argmax(-1).cpu()
            feats.append(lg.get(len(y)))
        acc.append((pred == y).numpy().astype(float))
    return np.concatenate(acc), torch.cat(feats).numpy()


def auroc(pos, neg):
    s = np.concatenate([pos, neg]); r = s.argsort().argsort() + 1
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def routing(fid, ftg, q):
    f_id, f_tg = np.log1p(fid.reshape(len(fid), -1)), np.log1p(ftg.reshape(len(ftg), -1))
    cal = np.zeros(len(f_id), bool); cal[np.random.default_rng(0).choice(len(f_id), len(f_id) // 2, replace=False)] = True
    med = np.median(f_id[cal], 0); iqr = np.subtract(*np.percentile(f_id[cal], [75, 25], 0)) + 1e-6
    sc = lambda f: np.abs((f - med) / iqr).mean(1)                   # noqa: E731
    tau = np.percentile(sc(f_id[cal]), 100 * q)
    s_id, s_tg = sc(f_id[~cal]), sc(f_tg)
    return dict(routed_id=100 * (s_id > tau).mean(), routed_target=100 * (s_tg > tau).mean(), auroc=auroc(s_tg, s_id),
                iqr_zero_frac=float((iqr <= 1e-6 + 1e-12).mean()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--n_calib", type=int, default=2000)
    p.add_argument("--q", type=float, default=0.8)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from camelyon import Classifier, MEAN, STD
    from domainbed_clip import load_splits, DomainImages, classnames, build_adapters
    from torchvision import transforms as T
    i = ES.info_of(a.run)
    model = Classifier(i.clip, n_cls=len(classnames(i.root, i.dataset)))
    for q in model.tower.parameters():
        q.requires_grad_(False)
    build_adapters(model, i)
    ES.load_sd(model, a.run)
    layers = lena_layers(model)
    tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(MEAN, STD)])
    tr, va, te = load_splits(i.root, i.dataset, i.target, i.holdout, i.split_seed, a.limit or i.limit)
    X.install_global(acc=True)
    res = {}
    acc_id, f_id = run_split(model, layers, DomainImages(va, tf), a.bs, a.workers)
    acc_tg, f_tg = run_split(model, layers, DomainImages(te, tf), a.bs, a.workers)
    res["standard"] = dict(acc_id=100 * acc_id.mean(), acc_target=100 * acc_tg.mean(), **routing(f_id, f_tg, a.q))
    X.EXT["qmatch"], X.EXT["qm_calib"] = True, True
    idx = np.random.RandomState(0).permutation(len(tr))[:a.n_calib]
    with torch.no_grad(), ES.autocast():
        for x, _, _ in DataLoader(Subset(DomainImages(tr, tf), idx), a.bs, num_workers=a.workers):
            model(x.cuda())
    X.qmatch_finalize(model)
    X.EXT["qm_calib"] = False
    acc_id, f_id = run_split(model, layers, DomainImages(va, tf), a.bs, a.workers)
    acc_tg, f_tg = run_split(model, layers, DomainImages(te, tf), a.bs, a.workers)
    res["r3_qmatch"] = dict(acc_id=100 * acc_id.mean(), acc_target=100 * acc_tg.mean(), **routing(f_id, f_tg, a.q))
    res["n"] = dict(id=len(va), target=len(te), calib=int(len(idx)))
    json.dump(res, open(os.path.join(a.out, "r3_diag.json"), "w"), indent=1)
    print("R3_DIAG", json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
