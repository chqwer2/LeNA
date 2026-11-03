"""R3 '--qt' signal diagnostic on existing LeNA checkpoints (eval only, no training).
For every LeNA layer and rank channel, u = A1 x per input: (1) split-half stability of the input's 16 u-quantiles
(R2 of half-A quantiles against half-B quantiles; checkerboard halves on square token grids, else alternating);
(2) separation of OOD from ID by the distance of Q_x to Q_ref = mean Q_x over a seed-0 half of the ID split:
  d_ref(x) = mean_{layer, channel, k} |Q_x - Q_ref| / sd_k(Q_ref)        (the --qt F statistic)
  d_z(x)   = mean |Q_x - Q_ref| / sd_items(ID calibration half)           (feature-wise z distance)
reported as AUROC(OOD vs ID eval half). Reuses eval_switch's task loaders (same items as the switch eval).

  python diag_qt.py --task camelyon --run runs/cam_lena_lr5e-5 --n 1000 --out runs/diag_qt_cam
"""
import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_switch as ES  # noqa: E402
from lena_qt import weighted_quantiles  # noqa: E402
from lena_ttr import halves  # noqa: E402

STATE = {"mri": False}


class QLog:
    def __init__(self, layers):
        self.layers = layers

    def __enter__(self):
        for L in self.layers:
            L.log_stats = True
        return self

    def get(self, b):
        r2s, qs = [], []
        for L in self.layers:
            u = L.last["u"].float()
            if u.shape[0] != b:                                          # several windows per image: pool tokens
                u = u.reshape(b, -1, u.shape[-1])
            w = torch.ones_like(u)
            q = weighted_quantiles(u, w)                                 # (b, r, 16)
            a = halves(u.shape[1], u.device)
            qa, qb = weighted_quantiles(u[:, a], w[:, a]), weighted_quantiles(u[:, ~a], w[:, ~a])
            sst = ((qb - qb.mean(-1, keepdim=True)) ** 2).sum(-1).clamp_min(1e-12)
            r2 = (1 - ((qa - qb) ** 2).sum(-1) / sst).mean(-1).clamp(-5, 1)           # (b,)
            r2s.append(r2); qs.append(q.flatten(1))
        STATE["nl"], STATE["r"] = len(self.layers), q.shape[1]
        f = torch.cat([torch.stack(r2s, 1), torch.cat(qs, 1)], 1).cpu()
        return torch.expm1(f) if STATE["mri"] else f                   # MRI path averages log1p over slices

    def __exit__(self, *a):
        for L in self.layers:
            L.log_stats = False; L.last = None


def auroc(pos, neg):
    s = np.concatenate([pos, neg]); r = s.argsort().argsort() + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["camelyon", "domainbed", "mri"], required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--mri_test", default="")
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ES.DEV["dev"] = "cuda"
    ES.FLog = QLog
    STATE["mri"] = a.task == "mri"
    args = SimpleNamespace(n=a.n, bs=a.bs, workers=2, limit=0, max_slices=0, mri_test=a.mri_test, clip_sets="val")
    out, idn = ES.TASKS[a.task](a.run, True, args)
    feats = {}
    for k, v in out.items():
        f = v["feat"]
        if STATE["mri"]:
            f = f.reshape(len(f), -1)
        feats[k] = f
    nid = len(feats[idn]); cal = np.zeros(nid, bool); cal[np.random.default_rng(0).choice(nid, nid // 2, replace=False)] = True
    nl, r = STATE["nl"], STATE["r"]                                   # feature = [R2 per layer, layers x r x 16 quantiles]
    res = dict(task=a.task, run=a.run, n_layers=nl, rank=r, splits={})
    Qid = feats[idn][cal, nl:].reshape(-1, nl, r, 16)
    qref = Qid.mean(0); sdk = qref.std(-1, keepdims=True) + 1e-6
    sdi = feats[idn][cal, nl:].std(0) + 1e-6
    d_ref = lambda f: (np.abs(f[:, nl:].reshape(-1, nl, r, 16) - qref) / sdk).mean((1, 2, 3))   # noqa: E731
    d_z = lambda f: (np.abs(f[:, nl:] - Qid.reshape(len(Qid), -1).mean(0)) / sdi).mean(1)     # noqa: E731
    id_eval = feats[idn][~cal]
    for k, f in feats.items():
        g = out[k].get("group")
        row = dict(n=len(f), split_half_r2_median=float(np.median(f[:, :nl])),
                   split_half_r2_by_layer=[float(x) for x in np.median(f[:, :nl], 0)],
                   d_ref_mean=float(d_ref(f).mean()), d_z_mean=float(d_z(f).mean()))
        if k != idn:
            row.update(auroc_d_ref=auroc(d_ref(f), d_ref(id_eval)), auroc_d_z=auroc(d_z(f), d_z(id_eval)))
            if g is not None and len(np.unique(g)) > 1:
                row["by_group"] = {str(gg): dict(n=int((g == gg).sum()), auroc_d_ref=auroc(d_ref(f[g == gg]), d_ref(id_eval)),
                                                 auroc_d_z=auroc(d_z(f[g == gg]), d_z(id_eval)))
                                   for gg in np.unique(g)}
        res["splits"][k] = row
    json.dump(res, open(os.path.join(a.out, "diag_qt.json"), "w"), indent=1)
    print("DIAG_QT", json.dumps({k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()
                                     if kk not in ("split_half_r2_by_layer",)} for k, v in res["splits"].items()}), flush=True)


if __name__ == "__main__":
    main()
