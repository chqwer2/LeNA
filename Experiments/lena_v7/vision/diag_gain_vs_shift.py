"""Per-patch evidence for "the gain follows the shift" on Camelyon17: for random patches of every hospital, the LeNA
fit statistic F and step s of every channel (logged in the forward pass), and the class probabilities of the LeNA and
LoRA checkpoints. The figure bins patches by a shift score computed from F.

  python diag_gain_vs_shift.py --lena runs/cam_lena_lr5e-5/best.pt --lora runs/cam_lora_lr1e-4/best.pt --out runs/diag_cam
"""
import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from lena_v7 import lena_layers  # noqa: E402
from camelyon import ParquetImages, MEAN, STD  # noqa: E402
from diag_camelyon import build, D  # noqa: E402
from torchvision import transforms as T  # noqa: E402


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lena", required=True); p.add_argument("--lora", required=True); p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=3000, help="random patches per hospital")
    a = p.parse_args(); os.makedirs(a.out, exist_ok=True); dev = "cuda"
    tf = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BILINEAR), T.ToTensor(), T.Normalize(MEAN, STD)])
    mle, mlo = build("lena", a.lena, dev), build("lora", a.lora, dev)
    layers = lena_layers(mle)
    rec = {k: [] for k in ("hosp", "y", "p_lena", "p_lora", "F", "s")}
    for split, hs in (("validation", (0, 3, 4, 1)), ("test", (2,))):
        ds = ParquetImages([os.path.join(D, f"{split}.pkl")], tf); rng = np.random.RandomState(0)
        for h in hs:
            idx = [i for i, c in enumerate(ds.cs) if c == h]; idx = list(rng.choice(idx, min(a.n, len(idx)), replace=False))
            for x, y, c in DataLoader(Subset(ds, idx), 128, num_workers=4):
                x = x.to(dev)
                for L in layers:
                    L.log_stats = True
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    ple = mle(x).float().softmax(-1)[:, 1]
                    plo = mlo(x).float().softmax(-1)[:, 1]
                rec["F"].append(torch.stack([L.last["F"].float().cpu() for L in layers], 1))   # (B, M, r)
                rec["s"].append(torch.stack([L.last["s"].float().cpu() for L in layers], 1))
                for L in layers:
                    L.log_stats = False; L.last = None
                rec["hosp"].append(c); rec["y"].append(y); rec["p_lena"].append(ple.cpu()); rec["p_lora"].append(plo.cpu())
            print("hospital", h, len(idx), flush=True)
    out = {k: torch.cat(v) for k, v in rec.items()}
    torch.save(out, os.path.join(a.out, "gain_vs_shift.pt")); print("DONE", {k: tuple(v.shape) for k, v in out.items()})


if __name__ == "__main__":
    main()
