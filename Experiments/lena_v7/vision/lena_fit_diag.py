"""Fit diagnostics of a trained MRI LeNA (mri_s0_lena): per module, the Stein factor s, the fraction of retained
knots, and how often tumour / brain / background tokens sit in rarely visited knots.

  python lena_fit_diag.py --ckpt runs/mri_s0_lena/best.pt --out lena_fit_diag_v1.csv
"""
import argparse
import types

import numpy as np
import torch
import torch.nn.functional as F

import mri_sam_seg as M
from lena_v7 import hat_index, lena_layers


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=20)
    a = p.parse_args()
    dev = "cuda"
    model = M.SamSeg(M.SAM, 224)
    M.build_adapters(model, types.SimpleNamespace(adapter="lena", r=8, alpha=16, mode="lena", targets=["qkv"], gate_drop=0.0))
    model.to(dev).eval()
    sd = torch.load(a.ckpt)
    model.load_state_dict({k: v.to(dev) for k, v in sd.items()}, strict=False)
    Ls = lena_layers(model)
    for L in Ls:
        L.log_stats = True
    gli = M.load_index(M.DATA, ["BraTS-2023-GLI"])                  # same split as the runs (seed 0)
    tr_all, _ = M.split_patients(gli, 0.2, 1)
    _, va = M.split_patients(tr_all, 0.1, 0)
    ssa = M.load_index(M.DATA, ["BraTS-2023-SSA"])
    rows = []
    for coh, vs in (("GLI-val", va[:a.n]), ("SSA", ssa[:a.n])):
        acc = {li: dict(s=[], vis=[], occ={c: [] for c in "tbg"}, rare={c: [] for c in "tbg"}) for li in range(len(Ls))}
        for v in vs:
            img = np.load(v["path"] + "/img.npy", mmap_mode="r")
            seg = np.load(v["path"] + "/seg.npy", mmap_mode="r")
            zs = np.array(v["brain_slices"])
            for z0 in range(0, len(zs), 40):
                zz = zs[z0:z0 + 40]
                x = torch.from_numpy(np.array(img[zz], dtype=np.float32)).to(dev)
                y = torch.from_numpy(np.array(seg[zz]) > 0).float().to(dev)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    model(x)
                xp, yp = M.pad224(x), F.pad(y, (16, 16, 16, 16))
                fg = F.adaptive_avg_pool2d((xp != M.BG["value"]).any(1, keepdim=True).float(), 14).flatten(1)
                tu = F.adaptive_avg_pool2d(yp[:, None], 14).flatten(1)
                cls = torch.where(tu >= 0.5, 0, torch.where(fg >= 0.5, 1, 2))          # t / b(rain) / g (background)
                for li, L in enumerate(Ls):
                    d = L.last
                    acc[li]["s"].append(d["s"].float().flatten().cpu())
                    acc[li]["vis"].append((d["p"].float() / L.K).flatten().cpu())
                    j, lam = hat_index(d["u"].float(), L.lo, L.hi, L.K)              # (B, N, r)
                    knot = (j + (lam > 0.5).long()).clamp(0, L.K - 1)
                    oh = F.one_hot(knot, L.K).float()                              # (B, N, r, K)
                    occ = oh.sum(1) / oh.shape[1]                                  # per input, per channel
                    tok_occ = (oh * occ[:, None]).sum(-1)                           # occupancy of the token's knot
                    for ci, c in enumerate("tbg"):
                        m = (cls == ci)[..., None].expand_as(tok_occ)
                        if m.any():
                            acc[li]["occ"][c].append(tok_occ[m].cpu())
                            acc[li]["rare"][c].append((tok_occ[m] < 0.02).float().cpu())
        for li in range(len(Ls)):
            s = torch.cat(acc[li]["s"]).numpy()
            r = dict(cohort=coh, layer=li, s_mean=s.mean(), s_p10=np.percentile(s, 10), s_p90=np.percentile(s, 90),
                     knots_retained=torch.cat(acc[li]["vis"]).mean().item())
            for c, nm in zip("tbg", ("tumour", "brain", "bg")):
                if acc[li]["occ"][c]:
                    r[f"knot_occ_{nm}"] = torch.cat(acc[li]["occ"][c]).mean().item()
                    r[f"rare_knot_frac_{nm}"] = torch.cat(acc[li]["rare"][c]).mean().item()
            rows.append({k: (round(float(v), 4) if not isinstance(v, str) else v) for k, v in r.items()})
            print(rows[-1], flush=True)
    keys = list(rows[0])
    with open(a.out, "w") as f:
        f.write(",".join(keys) + "\n" + "".join(",".join(str(r.get(k, "")) for k in keys) + "\n" for r in rows))


if __name__ == "__main__":
    main()
