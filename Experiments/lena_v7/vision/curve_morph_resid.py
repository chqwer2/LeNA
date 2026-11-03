"""Per-image fit residuals of LeNA on Camelyon17 ID patches under increasing stain shift (sweep_stain.py transform).

For each image, layer and channel: RMS residual of the second read v under (i) the curve fitted to this input,
(ii) the curve fitted to the same image without shift (frozen at sigma = 0), (iii) the learned prior curve; plus the
step s and the RMS change of the fitted curve's output relative to sigma = 0. Same subset and order as sweep_stain.py.

  python curve_morph_resid.py --n 3000 --out runs/curve_morph
"""
import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import lena_v7  # noqa: E402
from lena_v7 import lena_layers, hat_index, spline_eval  # noqa: E402
from camelyon import ParquetImages, MEAN, STD  # noqa: E402
from sweep_stain import SIGMAS, Perturbed, build  # noqa: E402
from torchvision import transforms as T  # noqa: E402


@torch.no_grad()
def run(model, x):
    layers = lena_layers(model)
    for L in layers:
        L.log_stats = True
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred = model(x).argmax(-1)
    out = []
    for L in layers:
        d = L.last
        out.append((d["u"].float()[:, 1:], d["v"].float()[:, 1:], d["theta"].float(), d["s"].float(), L.rho.float(), L))
        L.log_stats = False; L.last = None
    return pred, out


def rms(a):
    return a.pow(2).mean(1).sqrt()                                      # over tokens -> (B, r)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/mnt/scratch2/users/hchen/data/lena_v7/camelyon17/pkl")
    p.add_argument("--n", type=int, default=3000); p.add_argument("--bs", type=int, default=100)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    lena_v7.set_triton(False)
    base = ParquetImages([os.path.join(a.data, "validation.pkl")], None, keep={0, 3, 4}, limit=a.n)
    tf = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BILINEAR), T.ToTensor(), T.Normalize(MEAN, STD)])
    m = build("lena", "runs/cam_lena_lr5e-5/best.pt", "openai/clip-vit-base-patch16")
    keys = ("r_fit", "r_frozen", "r_prior", "s", "d_out")
    res = dict(sigmas=SIGMAS, y=np.array(base.ys), centre=np.array(base.cs), pred=[], **{k: [] for k in keys})
    th0 = {}                                                            # (batch, layer) -> theta at sigma 0
    for li, s in enumerate(SIGMAS):
        dl = DataLoader(Perturbed(base, s, tf), a.bs, num_workers=4)
        acc = {k: [] for k in keys + ("pred",)}
        for bi, (x, _) in enumerate(dl):
            pred, out = run(m, x.cuda())
            acc["pred"].append(pred.cpu())
            per = {k: [] for k in keys}
            for l, (u, v, th, st, rho, L) in enumerate(out):
                if li == 0:
                    th0[(bi, l)] = (th.clone(), None)
                j, lam = hat_index(u, L.lo, L.hi, L.K)
                c = spline_eval(th.unsqueeze(1), j, lam)
                c0 = spline_eval(th0[(bi, l)][0].unsqueeze(1), j, lam)
                cp = spline_eval(rho, j, lam)
                if li == 0:
                    th0[(bi, l)] = (th0[(bi, l)][0], c.clone())
                per["r_fit"].append(rms(c - v)); per["r_frozen"].append(rms(c0 - v)); per["r_prior"].append(rms(cp - v))
                per["s"].append(st)
                ref = th0[(bi, l)][1]
                per["d_out"].append(rms(c - ref))
            for k in keys:
                acc[k].append(torch.stack(per[k], 1).half().cpu())   # (B, 72, r)
        for k in keys + ("pred",):
            res[k].append(torch.cat(acc[k]))
        ok = (res["pred"][-1].numpy() == res["y"]).mean()
        print(f"sigma {s}: acc {ok:.4f} r_fit {res['r_fit'][-1].float().median():.3f} "
              f"r_frozen {res['r_frozen'][-1].float().median():.3f} r_prior {res['r_prior'][-1].float().median():.3f} "
              f"mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
    for k in keys + ("pred",):
        res[k] = torch.stack(res[k])                                    # (levels, B, 72, r)
    torch.save(res, os.path.join(a.out, "curve_morph_resid.pt"))
    print("DONE")


if __name__ == "__main__":
    main()
