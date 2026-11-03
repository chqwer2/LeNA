"""S4 screen (eval only, H14): is the per-input fit's gain over rho an affine re-centring/rescaling of the shared
curve, or shape? Forward-only runtime patch of lena_v7.fit_and_step (this process only; lena_v7.py untouched).

Everything is computed from the input's spline statistics (n = tridiagonal Gram N, m, e, ntok). Hat functions sum
to one, so the constant curve is the all-ones knot vector 1, and for curves f, g on the knot grid the token sums are
sum_i f(u_i) g(u_i) = f'Ng, sum_i f(u_i) v_i = f.m. S = span{rho, 1} (curves a*rho(u) + b).
  plain   (a) usual LeNA: theta = rho + s Delta                                     (control; must reproduce)
  affine  (b) theta = rho + s (a*rho + b*1 - rho), (a, b) = least squares of v on [rho(u_i), 1] over the input's
              tokens; s = the same Stein shrink as (a) (0 where (a) is not identifiable)
  fixed   (c) theta = rho (s = 0)
  shape   (d) theta = theta_a - P_S theta_a + rho  (P_S = N-orthogonal projection onto S: the fitted curve with its
              affine part on the tokens removed, then rho added back)
Logged for every fit (from (a)'s theta, whatever variant is applied): fraction of the token variance of the fitted
departure delta = theta_a - rho explained by its affine part P_S delta, variance-weighted over inputs/channels:
  sum var_tok(P_S delta) / sum var_tok(delta),  var_tok(f) = f'Nf/n - (1'Nf/n)^2.

  python screen_affine.py --variant affine --task camelyon --lena runs/cam_lena_lr5e-5 --out runs/screens/s4_v1/cam_affine
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import eval_switch as ES  # noqa: E402
import lena_v7 as LV  # noqa: E402

ORIG = LV.fit_and_step
G = {"variant": "plain", "n": 0, "num": 0.0, "den": 0.0, "frac_mean": 0.0, "s": 0.0}


def proj_S(rho, f, diag, off):
    """N-orthogonal projection of curve(s) f onto span{rho, 1}; returns P f (..., K)."""
    one = torch.ones_like(f)
    Nr, N1 = LV.tri_mv(diag, off, rho), LV.tri_mv(diag, off, one)
    g11, g12, g22 = (rho * Nr).sum(-1), (rho * N1).sum(-1), (one * N1).sum(-1)
    b1, b2 = (Nr * f).sum(-1), (N1 * f).sum(-1)
    ridge = 1e-6 * (g11 + g22).clamp_min(1e-12)
    g11, g22 = g11 + ridge, g22 + ridge
    det = (g11 * g22 - g12 * g12).clamp_min(1e-20)
    a = (g22 * b1 - g12 * b2) / det
    b = (g11 * b2 - g12 * b1) / det
    return a.unsqueeze(-1) * rho + b.unsqueeze(-1) * one


def var_tok(f, diag, off, ntok):
    n = ntok.clamp_min(1)
    return ((f * LV.tri_mv(diag, off, f)).sum(-1) / n - ((LV.tri_mv(diag, off, torch.ones_like(f)) * f).sum(-1) / n) ** 2
            ).clamp_min(0)


def fit_and_step(diag, off, m, e, ntok, rho, omega, tau, mode):
    theta, d = ORIG(diag, off, m, e, ntok, rho, omega, tau, mode)
    if mode != "lena":
        return theta, d
    rb = rho.expand_as(m)
    delta = theta - rb
    pd = proj_S(rb, delta, diag, off)
    vd, vp = var_tok(delta, diag, off, ntok), var_tok(pd, diag, off, ntok)
    G["n"] += 1; G["num"] += vp.sum().item(); G["den"] += vd.sum().item(); G["s"] += d["s"].float().mean().item()
    ok = vd > 1e-12
    G["frac_mean"] += ((vp / vd.clamp_min(1e-12)).clamp(0, 1)[ok].mean().item() if ok.any() else 0.0)
    v = G["variant"]
    if v == "plain":
        return theta, d
    if v == "fixed":
        return rb, d
    if v == "shape":
        return theta - proj_S(rb, theta, diag, off) + rb, d
    # affine: LS of v on [rho(u), 1]  ==  N-projection of the unconstrained LS curve onto S, i.e. solve with rhs m
    one = torch.ones_like(m)
    Nr, N1 = LV.tri_mv(diag, off, rb), LV.tri_mv(diag, off, one)
    g11, g12, g22 = (rb * Nr).sum(-1), (rb * N1).sum(-1), (one * N1).sum(-1)
    b1, b2 = (rb * m).sum(-1), m.sum(-1)
    ridge = 1e-6 * (g11 + g22).clamp_min(1e-12)
    g11, g22 = g11 + ridge, g22 + ridge
    det = (g11 * g22 - g12 * g12).clamp_min(1e-20)
    a = (g22 * b1 - g12 * b2) / det
    b = (g11 * b2 - g12 * b1) / det
    aff = a.unsqueeze(-1) * rb + b.unsqueeze(-1) * one
    return rb + d["s"].unsqueeze(-1) * (aff - rb), d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["plain", "affine", "fixed", "shape"], required=True)
    p.add_argument("--task", choices=list(ES.TASKS), required=True)
    p.add_argument("--lena", required=True)
    p.add_argument("--n", type=int, default=3000)
    p.add_argument("--clip_sets", default="val,v2,sketch,a,r")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max_slices", type=int, default=0)
    p.add_argument("--mri_test", default="")
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    ES.DEV["dev"] = a.device
    G["variant"] = a.variant
    LV.fit_and_step = fit_and_step
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    res, idn = ES.TASKS[a.task](a.lena, False, a)
    np.savez_compressed(os.path.join(a.out, "items.npz"),
                        **{f"{k}|{f}": v[f] for k, v in res.items() for f in ("metric", "group") if v.get(f) is not None})
    n = max(1, G["n"])
    summ = dict(args=vars(a), splits={k: dict(n=len(v["metric"]), mean=100 * float(v["metric"].mean()))
                                      for k, v in res.items()},
                affine_var_frac=G["num"] / max(G["den"], 1e-12), affine_var_frac_mean=G["frac_mean"] / n,
                mean_s=G["s"] / n, n_fits=G["n"], wall_s=time.time() - t0)
    json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print("SUMMARY", json.dumps({k: v for k, v in summ.items() if k != "args"}), flush=True)


if __name__ == "__main__":
    main()
