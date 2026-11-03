"""S3 screen (eval only): R^2-gated LeNA fit. Forward-only runtime patch of lena_v7 (this process only; lena_v7.py is
not edited). For every LeNA layer, input and rank channel (non-causal fits only):
  1. split the input's tokens into two interleaved halves A/B: checkerboard over the square token grid (a leading
     CLS token, N = g*g + 1, goes to A); otherwise alternating positions;
  2. Stein fit (the layer's own mode, here "lena") on A, scored on B, and on B scored on A;
     R2_cv = 1 - [SSE_B(theta_A) + SSE_A(theta_B)] / [SSE_B(rho) + SSE_A(rho)]   (held-out, relative to the shared curve)
     SSE_X(theta) = e_X - 2 theta.m_X + theta' n_X theta from the spline statistics of half X;
  3. theta = usual Stein theta of all tokens if R2_cv > 0, else theta = rho (fixed curve) for that input/channel.
Then runs eval_switch's task loader for the LeNA model only (same items, order and batch size as runs/ext_none_*),
and writes per-item metrics plus the gated fraction.

  python screen_r2gate.py --task camelyon --lena runs/cam_lena_lr5e-5 --out runs/screens/s3/cam [--n 3000 --bs 64]
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

ORIG = {"fit_and_step": LV.fit_and_step, "pooled_stats": LV.pooled_stats}
G = {"halves": None, "on": True, "n": 0, "gated": 0.0, "r2": 0.0, "s": 0.0}


def half_mask(N, device):
    g = int(round(N ** 0.5))
    if g * g == N or (g * g == N - 1):
        cls = N - g * g
        i = torch.arange(g, device=device)
        cb = ((i[:, None] + i[None, :]) % 2 == 0).flatten()
        return torch.cat([torch.ones(cls, dtype=torch.bool, device=device), cb])
    return torch.arange(N, device=device) % 2 == 0


def pooled_stats(u, v, w, lo, hi, K):
    out = ORIG["pooled_stats"](u, v, w, lo, hi, K)
    if G["on"]:
        a = half_mask(u.shape[1], u.device).to(w.dtype)[None, :, None]
        sa = ORIG["pooled_stats"](u, v, w * a, lo, hi, K)[2:]
        sb = ORIG["pooled_stats"](u, v, w * (1 - a), lo, hi, K)[2:]
        G["halves"] = (sa, sb, out[2].shape)
    return out


def sse(theta, diag, off, m, e):
    return e - 2 * (theta * m).sum(-1) + (theta * LV.tri_mv(diag, off, theta)).sum(-1)


def fit_and_step(diag, off, m, e, ntok, rho, omega, tau, mode):
    h, G["halves"] = G["halves"], None
    theta, d = ORIG["fit_and_step"](diag, off, m, e, ntok, rho, omega, tau, mode)
    if not G["on"] or mode != "lena" or h is None or h[2] != diag.shape:
        return theta, d
    sa, sb, _ = h
    ta, _ = ORIG["fit_and_step"](*sa, rho, omega, tau, mode)
    tb, _ = ORIG["fit_and_step"](*sb, rho, omega, tau, mode)
    rb = rho.expand_as(m)
    num = sse(ta, *sb[:4]) + sse(tb, *sa[:4])
    den = sse(rb, *sb[:4]) + sse(rb, *sa[:4])
    r2 = 1 - num / den.clamp_min(1e-12)
    keep = r2 > 0
    theta = torch.where(keep.unsqueeze(-1), theta, rb)
    G["n"] += 1; G["gated"] += (~keep).float().mean().item(); G["r2"] += r2.clamp(-5, 1).float().mean().item()
    G["s"] += d["s"].float().mean().item()
    return theta, d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=list(ES.TASKS), required=True)
    p.add_argument("--lena", required=True)
    p.add_argument("--off", action="store_true", help="control: patch installed but gate disabled (= plain LeNA)")
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
    G["on"] = not a.off
    LV.fit_and_step, LV.pooled_stats = fit_and_step, pooled_stats
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    if a.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    res, idn = ES.TASKS[a.task](a.lena, False, a)
    np.savez_compressed(os.path.join(a.out, "items.npz"),
                        **{f"{k}|{f}": v[f] for k, v in res.items() for f in ("metric", "group") if v.get(f) is not None})
    n = max(1, G["n"])
    summ = dict(args=vars(a), splits={k: dict(n=len(v["metric"]), mean=100 * float(v["metric"].mean()))
                                      for k, v in res.items()},
                frac_gated=G["gated"] / n, mean_r2cv=G["r2"] / n, mean_s=G["s"] / n, n_fits=G["n"],
                peak_gb=torch.cuda.max_memory_allocated() / 2**30 if a.device == "cuda" else 0,
                wall_s=time.time() - t0)
    json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print("SUMMARY", json.dumps({k: v for k, v in summ.items() if k != "args"}), flush=True)


if __name__ == "__main__":
    main()
