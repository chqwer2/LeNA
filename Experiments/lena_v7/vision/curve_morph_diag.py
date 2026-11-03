"""Fitted LeNA curves on the same Camelyon17 ID patches under increasing stain shift (the sweep_stain.py transform).

Per stain level sigma and image: step s, fit size F, fitted ordinates theta, per-channel code statistics of u, and
the prediction; the 8 first images also keep their full codes (u, v) and perturbed RGB patches.

  python curve_morph_diag.py --n 512 --out runs/curve_morph
"""
import argparse
import io
import os
import sys

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import lena_v7  # noqa: E402
from lena_v7 import lena_layers, LeNALinear  # noqa: E402
from camelyon import ParquetImages, MEAN, STD  # noqa: E402
from sweep_stain import SIGMAS, Perturbed, build, stain_perturb  # noqa: E402
from torchvision import transforms as T  # noqa: E402


@torch.no_grad()
def run(model, x):
    layers = lena_layers(model)
    for L in layers:
        L.log_stats = True
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred = model(x).argmax(-1).cpu()
    rec = {k: [] for k in ("s", "F", "theta", "umean", "ustd", "uq05", "uq95", "u", "v")}
    for L in layers:
        d = L.last; u = d["u"].float()[:, 1:]                       # (B, N, r), drop CLS
        rec["s"].append(d["s"].float()); rec["F"].append(d["F"].float()); rec["theta"].append(d["theta"].float().half())
        rec["umean"].append(u.mean(1)); rec["ustd"].append(u.std(1))
        q = torch.quantile(u, torch.tensor([0.05, 0.95], device=u.device), dim=1)
        rec["uq05"].append(q[0]); rec["uq95"].append(q[1])
        rec["u"].append(d["u"].half()); rec["v"].append(d["v"].half())
        L.log_stats = False; L.last = None
    return pred, {k: torch.stack(v, 1).cpu() for k, v in rec.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/mnt/scratch2/users/hchen/data/lena_v7/camelyon17/pkl")
    p.add_argument("--n", type=int, default=512); p.add_argument("--bs", type=int, default=64)
    p.add_argument("--n_show", type=int, default=8); p.add_argument("--out", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    lena_v7.set_triton(False)
    base = ParquetImages([os.path.join(a.data, "validation.pkl")], None, keep={0, 3, 4}, limit=a.n)
    tf = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BILINEAR), T.ToTensor(), T.Normalize(MEAN, STD)])
    m = build("lena", "runs/cam_lena_lr5e-5/best.pt", "openai/clip-vit-base-patch16")
    names = [n for n, mod in m.tower.named_modules() if isinstance(mod, LeNALinear)]
    res = dict(names=names, sigmas=SIGMAS, y=np.array(base.ys), centre=np.array(base.cs),
               rho=torch.stack([mod.rho.detach().float().cpu() for mod in m.modules() if isinstance(mod, LeNALinear)]),
               lohi=(lena_layers(m)[0].lo, lena_layers(m)[0].hi), levels=[])
    for s in SIGMAS:
        dl = DataLoader(Perturbed(base, s, tf), a.bs, num_workers=4)
        acc = {k: [] for k in ("pred", "s", "F", "theta", "umean", "ustd", "uq05", "uq95")}; show = {}
        for bi, (x, y) in enumerate(dl):
            pred, r = run(m, x.cuda())
            acc["pred"].append(pred)
            for k in ("s", "F", "theta", "umean", "ustd", "uq05", "uq95"):
                acc[k].append(r[k])
            if bi == 0:
                show = dict(u=r["u"][:a.n_show], v=r["v"][:a.n_show])
        lev = {k: torch.cat(v) for k, v in acc.items()}
        lev["show"] = show
        lev["img"] = np.stack([np.array(stain_perturb(Image.open(io.BytesIO(base.imgs[i])).convert("RGB"), s, i))
                               for i in range(a.n_show)])
        res["levels"].append(lev)
        print(f"sigma {s}: acc {(lev['pred'].numpy() == res['y']).mean():.4f}, median s {lev['s'].median():.3f}, "
              f"mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
    torch.save(res, os.path.join(a.out, "curve_morph.pt"))
    print("DONE")


if __name__ == "__main__":
    main()
