"""Per-patch evidence for the shift switch on Camelyon17: one LeNA model trained with --gate_drop (shared-curve path
trained too) evaluated per patch with the per-input curve (learned) and with the shared curve only (fixed, s = 0),
plus the baseline LeNA and LoRA on the same patches. Same sampling as diag_gain_vs_shift.py (3000 random patches per
hospital, RandomState(0) per split). F and s of every channel are logged in the learned passes; the shift score and the
switch are computed offline from F.

  python diag_gate_switch.py --gate runs/cam_lena_gate05_lr5e-5/best.pt --lena runs/cam_lena_lr5e-5/best.pt \
      --lora runs/cam_lora_lr1e-4/best.pt --out runs/diag_gate_switch
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from lena_v7 import lena_layers, lena_override  # noqa: E402
from camelyon import ParquetImages, MEAN, STD  # noqa: E402
from diag_camelyon import build, D  # noqa: E402
from torchvision import transforms as T  # noqa: E402


def logged(m, x):
    """Class-1 probability with the per-input curve, and F, s of every LeNA channel (B, M, r)."""
    layers = lena_layers(m)
    for L in layers:
        L.log_stats = True
    with torch.autocast("cuda", dtype=torch.bfloat16):
        p = m(x).float().softmax(-1)[:, 1]
    F_ = torch.stack([L.last["F"].float().cpu() for L in layers], 1)
    s_ = torch.stack([L.last["s"].float().cpu() for L in layers], 1)
    for L in layers:
        L.log_stats = False; L.last = None
    return p.cpu(), F_, s_


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gate", required=True); p.add_argument("--lena", required=True); p.add_argument("--lora", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=3000, help="random patches per hospital")
    a = p.parse_args(); os.makedirs(a.out, exist_ok=True); dev = "cuda"; t0 = time.time()
    tf = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BILINEAR), T.ToTensor(), T.Normalize(MEAN, STD)])
    mg, mle, mlo = build("lena", a.gate, dev), build("lena", a.lena, dev), build("lora", a.lora, dev)
    keys = ("hosp", "y", "idx", "p_gate_learned", "p_gate_fixed", "p_lena", "p_lora", "F_gate", "s_gate", "F_lena", "s_lena")
    rec = {k: [] for k in keys}
    for split, hs in (("validation", (0, 3, 4, 1)), ("test", (2,))):
        ds = ParquetImages([os.path.join(D, f"{split}.pkl")], tf); rng = np.random.RandomState(0)
        for h in hs:
            idx = [i for i, c in enumerate(ds.cs) if c == h]; idx = list(rng.choice(idx, min(a.n, len(idx)), replace=False))
            rec["idx"].append(torch.tensor(idx))
            for x, y, c in DataLoader(Subset(ds, idx), 128, num_workers=4):
                x = x.to(dev)
                pgl, Fg, sg = logged(mg, x)
                with lena_override(mode="fixed"), torch.autocast("cuda", dtype=torch.bfloat16):
                    pgf = mg(x).float().softmax(-1)[:, 1].cpu()
                pl, Fl, sl = logged(mle, x)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    plo = mlo(x).float().softmax(-1)[:, 1].cpu()
                for k, v in zip(keys, (c, y, None, pgl, pgf, pl, plo, Fg, sg, Fl, sl)):
                    if v is not None:
                        rec[k].append(v)
            m = slice(-len(idx), None)
            yy = torch.cat(rec["y"])[m]
            acc = {k: round(((torch.cat(rec[k])[m] > .5).long() == yy).float().mean().item(), 4)
                   for k in ("p_gate_learned", "p_gate_fixed", "p_lena", "p_lora")}
            print("hospital", h, len(idx), acc, f"{time.time() - t0:.0f}s", flush=True)
    out = {k: torch.cat(v) for k, v in rec.items()}
    for k in ("F_gate", "s_gate", "F_lena", "s_lena"):
        assert torch.isfinite(out[k]).all(), k
    torch.save(out, os.path.join(a.out, "gate_switch.pt")); print("DONE", {k: tuple(v.shape) for k, v in out.items()})


if __name__ == "__main__":
    main()
