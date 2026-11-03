"""Per-input evidence that LeNA's design works on real Camelyon17 patches (fig_design_live).

For random patches of every hospital (ID 0/3/4, OOD val 1, OOD test 2) this records, inference only:
  * P(tumour) of LoRA (lr 1e-4), LeNA (cam_lena_lr5e-5; learned / s=0 shared curve) and LeNA-S
    (cam_K05_R1_omega100: R1 design-effect step + learned omega; learned / s=0 / s=1)
  * per (module, channel) step s, fit statistic F and retained knots p of LeNA and LeNA-S
and, for a few patches per hospital and outcome category (LoRA wrong & LeNA-S right, etc.), the tokens (u, v), the
applied curve theta, the unshrunk fit theta_plain (s=1) and the raw patch image.

  python diag_design_live.py --out runs/diag_design_live --n 800
"""
import argparse
import io
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import lena_v7  # noqa: E402
from lena_v7 import lena_layers, LeNALinear  # noqa: E402
import lena_mri_ext  # noqa: E402
from camelyon import ParquetImages, MEAN, STD  # noqa: E402
from diag_camelyon import build, D  # noqa: E402
from torchvision import transforms as T  # noqa: E402

CATS = ("fixS", "fixL", "hurtS", "hurtL", "ok", "wrong")


def fwd(m, x, mode=None, log=False, neff=False, dev="cuda"):
    """P(tumour) and (optionally) per-layer diagnostics. neff toggles the R1 design-effect step (global patch)."""
    lena_mri_ext.EXT["neff"] = neff
    layers = lena_layers(m)
    for L in layers:
        L.log_stats = log
    with lena_v7.lena_override(mode=mode), torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
        p = m(x).float().softmax(-1)[:, 1]
    d = None
    if log:
        d = {k: torch.stack([L.last[k].float() for L in layers], 1).cpu() for k in layers[0].last
             if k in ("s", "F", "p", "theta", "u", "v")}
    for L in layers:
        L.log_stats = False; L.last = None
    lena_mri_ext.EXT["neff"] = False
    return p.cpu(), d


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", default="runs/cam_lora_lr1e-4/best.pt")
    ap.add_argument("--lena", default="runs/cam_lena_lr5e-5/best.pt")
    ap.add_argument("--lenaS", default="runs/cam_K05_R1_omega100/best.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=800, help="random patches per hospital")
    ap.add_argument("--per_cat", type=int, default=6)
    ap.add_argument("--bs", type=int, default=100)
    ap.add_argument("--dev", default="cuda")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); dev = a.dev; t0 = time.time()
    lena_v7.set_triton(False)
    mlo = build("lora", a.lora, dev); mle = build("lena", a.lena, dev); mls = build("lena", a.lenaS, dev)
    lena_mri_ext.install(mls, neff=True)                         # patches module functions; gated by EXT["neff"]
    lena_mri_ext.EXT["acc_eval"] = None; lena_mri_ext.EXT["deff_scale"] = 1.0; lena_mri_ext.EXT["neff"] = False
    names = [n for n, mod in mle.tower.named_modules() if isinstance(mod, LeNALinear)]
    tf = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BILINEAR), T.ToTensor(), T.Normalize(MEAN, STD)])
    P = {k: [] for k in ("hosp", "y", "idx", "lora", "lena", "lena_fixed", "lenaS", "lenaS_fixed", "lenaS_plain",
                         "s_lena", "F_lena", "p_lena", "s_lenaS", "F_lenaS", "p_lenaS")}
    sets = {}
    for split, hs in (("validation", (0, 3, 4, 1)), ("test", (2,))):
        ds = ParquetImages([os.path.join(D, f"{split}.pkl")], tf); rng = np.random.RandomState(0)
        for h in hs:
            idx = [i for i, c in enumerate(ds.cs) if c == h]
            idx = [int(i) for i in rng.choice(idx, min(a.n, len(idx)), replace=False)]
            sets[h] = (ds, idx)
            for x, y, c in DataLoader(Subset(ds, idx), a.bs, num_workers=4):
                x = x.to(dev)
                P["lora"].append(fwd(mlo, x, dev=dev)[0])
                pe, de = fwd(mle, x, log=True, dev=dev)
                P["lena"].append(pe); P["lena_fixed"].append(fwd(mle, x, mode="fixed", dev=dev)[0])
                ps, dS = fwd(mls, x, log=True, neff=True, dev=dev)
                P["lenaS"].append(ps); P["lenaS_fixed"].append(fwd(mls, x, mode="fixed", dev=dev)[0])
                P["lenaS_plain"].append(fwd(mls, x, mode="plain", dev=dev)[0])
                for k in ("s", "F", "p"):
                    P[f"{k}_lena"].append(de[k].half()); P[f"{k}_lenaS"].append(dS[k].half())
                P["hosp"].append(c); P["y"].append(y)
            P["idx"].append(torch.tensor(idx))
            print(f"hospital {h}: {len(idx)} patches, {time.time() - t0:.0f}s", flush=True)
    pop = {k: torch.cat(v) for k, v in P.items()}
    y = pop["y"].bool()
    for k in ("lora", "lena", "lena_fixed", "lenaS", "lenaS_fixed", "lenaS_plain"):
        ok = (pop[k] > 0.5) == y
        print("ACC", k, {h: round(ok[pop["hosp"] == h].float().mean().item(), 4) for h in (0, 3, 4, 1, 2)}, flush=True)
    # ---- case selection per hospital and outcome category
    ok = {k: ((pop[k] > 0.5) == y) for k in ("lora", "lena", "lenaS")}
    cat_mask = dict(fixS=~ok["lora"] & ok["lenaS"], fixL=~ok["lora"] & ok["lena"], hurtS=ok["lora"] & ~ok["lenaS"],
                    hurtL=ok["lora"] & ~ok["lena"], ok=ok["lora"] & ok["lena"] & ok["lenaS"],
                    wrong=~ok["lora"] & ~ok["lena"] & ~ok["lenaS"])
    rng = np.random.RandomState(1); cases = []; seen = set()
    for h in (0, 3, 4, 1, 2):
        for cn in CATS:
            rows = torch.nonzero(cat_mask[cn] & (pop["hosp"] == h)).flatten().tolist()
            for r in (list(rng.choice(rows, min(a.per_cat, len(rows)), replace=False)) if rows else []):
                if int(r) not in seen:
                    seen.add(int(r)); cases.append((int(r), h, cn))
    print("cases", len(cases), {cn: sum(c[2] == cn for c in cases) for cn in CATS}, flush=True)
    CS = []
    for i0 in range(0, len(cases), 20):
        chunk = cases[i0:i0 + 20]
        x = []; imgs = []
        for r, h, cn in chunk:
            ds, _ = sets[h]; di = int(pop["idx"][r]); x.append(ds[di][0])
            imgs.append(np.array(Image.open(io.BytesIO(ds.imgs[di])).convert("RGB")))
        x = torch.stack(x).to(dev)
        _, de = fwd(mle, x, log=True, dev=dev); _, dep = fwd(mle, x, mode="plain", log=True, dev=dev)
        _, dS = fwd(mls, x, log=True, neff=True, dev=dev); _, dSp = fwd(mls, x, mode="plain", log=True, dev=dev)
        for k, (r, h, cn) in enumerate(chunk):
            CS.append(dict(row=r, hosp=h, cat=cn, y=int(pop["y"][r]), img=imgs[k],
                           **{f"{q}_{m}": float(pop[m][r]) for q in ("p",) for m in
                              ("lora", "lena", "lena_fixed", "lenaS", "lenaS_fixed", "lenaS_plain")},
                           lena=dict(u=de["u"][k].half(), v=de["v"][k].half(), theta=de["theta"][k],
                                     theta_plain=dep["theta"][k], s=de["s"][k], F=de["F"][k], p=de["p"][k]),
                           lenaS=dict(u=dS["u"][k].half(), v=dS["v"][k].half(), theta=dS["theta"][k],
                                      theta_plain=dSp["theta"][k], s=dS["s"][k], F=dS["F"][k], p=dS["p"][k])))
    rho = dict(lena=torch.stack([L.rho.detach().float().cpu() for L in lena_layers(mle)]),
               lenaS=torch.stack([L.rho.detach().float().cpu() for L in lena_layers(mls)]))
    omega = torch.stack([L.omega.detach().float().cpu() for L in lena_layers(mls)])
    torch.save(dict(names=names, pop=pop, cases=CS, rho=rho, omega_lenaS=omega, ckpt=dict(lora=a.lora, lena=a.lena,
                    lenaS=a.lenaS), n_per_hosp=a.n, grid=(-3.0, 3.0, 16)), os.path.join(a.out, "design_live.pt"))
    print(f"DONE {len(CS)} cases, {pop['y'].numel()} patches, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
