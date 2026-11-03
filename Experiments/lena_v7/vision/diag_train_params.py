"""CPU diagnostic: how far each LeNA parameter group moved from its initialization, per run, vs LoRA.

Init (lena_v7.LeNALinear): A1, A2 xavier_uniform (E||A||_F^2 = r*d_in*2/(r+d_in)), B = 0, rho = linspace(lo,hi,K)
repeated per rank (identity curve), omega = 0. LoRA: A kaiming/xavier-type random, B = 0.
Prints, per run: median over layers of ||B||, ||A1||/E_init||A1||, ||A2||/E_init, ||rho - rho0||/||rho0||, omega,
and the max knot deviation. Also MRI curves.npz trajectories if present.
"""
import glob
import json
import os
import sys

import numpy as np
import torch

RUNS = sys.argv[1:]


def group(sd):
    layers = {}
    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        base, _, leaf = k.rpartition(".")
        layers.setdefault(base, {})[leaf] = v.float()
    return {b: d for b, d in layers.items() if "B" in d or "lora_B" in d}


def stats(path):
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    L = group(sd)
    out = dict(n_layers=len(L), keys=sorted({kk for d in L.values() for kk in d}))
    rows = []
    for b, d in L.items():
        Bm = d.get("B", d.get("lora_B"))
        A1 = d.get("A1", d.get("A", d.get("lora_A")))
        r, d_in = A1.shape if A1 is not None and A1.shape[0] < A1.shape[1] else (A1.shape[1], A1.shape[0]) if A1 is not None else (0, 0)
        e_init = np.sqrt(r * d_in * 2.0 / (r + d_in)) if r else np.nan
        row = dict(B=Bm.norm().item(), A1=(A1.norm().item() / e_init) if A1 is not None else np.nan)
        if "A2" in d:
            row["A2"] = d["A2"].norm().item() / e_init
        if "rho" in d:
            rho = d["rho"]; K = rho.shape[-1]
            rho0 = torch.linspace(-3, 3, K).expand_as(rho)
            row["rho_rel"] = ((rho - rho0).norm() / rho0.norm()).item()
            row["rho_maxdev"] = (rho - rho0).abs().max().item()
        if "omega" in d:
            row["omega"] = d["omega"].item()
        rows.append(row)
    for k in rows[0]:
        vals = np.array([r_[k] for r_ in rows if k in r_])
        out[k] = dict(med=float(np.median(vals)), p10=float(np.percentile(vals, 10)), p90=float(np.percentile(vals, 90)))
    return out


for run in RUNS:
    ck = [p for p in (f"{run}/best.pt", f"{run}/adapter.pt") if os.path.exists(p)]
    if not ck:
        print(run, "NO CKPT"); continue
    s = stats(ck[0])
    print("RUN", run, json.dumps({k: (v if not isinstance(v, dict) else {a: round(b, 4) for a, b in v.items()}) for k, v in s.items()}), flush=True)
    cv = f"{run}/curves.npz"
    if os.path.exists(cv):
        z = np.load(cv, allow_pickle=True)
        print("CURVES", run, {k: (z[k].shape, z[k].dtype.str) for k in z.files}, flush=True)
        for k in z.files:
            if z[k].dtype != object and z[k].ndim == 1 and z[k].size <= 40:
                print("  ", k, np.round(z[k], 4).tolist())
    rj = f"{run}/result.json"
    if os.path.exists(rj):
        r = json.load(open(rj))
        h = r.get("history") or r.get("hist")
        if isinstance(h, list) and h and isinstance(h[0], dict):
            print("HIST", run, [(e.get("step", e.get("epoch")), round(e.get("id_val", float("nan")), 4)) for e in h], flush=True)
