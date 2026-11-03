"""Training dynamics of LeNA on Camelyon17: for each saved checkpoint, on fixed random patches of the training
hospitals and of the test hospital H2: mean step s, share of kept knots, mean |theta - rho| (input-specific
deviation of the curve), its spread across patches, and accuracy.
  python diag_dynamics.py --run runs/cam_lena_dyn --n 300
"""
import argparse, glob, os, re, sys
import numpy as np, torch
from torch.utils.data import DataLoader, Subset
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from lena_v7 import lena_layers  # noqa
from camelyon import ParquetImages, MEAN, STD  # noqa
from diag_camelyon import build, D  # noqa
from torchvision import transforms as T  # noqa


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(); p.add_argument("--run", required=True); p.add_argument("--n", type=int, default=300)
    a = p.parse_args(); dev = "cuda"
    tf = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BILINEAR), T.ToTensor(), T.Normalize(MEAN, STD)])
    sets = {}
    for split, hs, tag in (("validation", (0, 3, 4), "train"), ("test", (2,), "H2")):
        ds = ParquetImages([os.path.join(D, f"{split}.pkl")], tf); rng = np.random.RandomState(0)
        idx = [i for i, c in enumerate(ds.cs) if c in hs]; sets[tag] = (ds, list(rng.choice(idx, a.n, replace=False)))
    ck = sorted(glob.glob(os.path.join(a.run, "ckpt_*.pt")), key=lambda f: int(re.findall(r"(\d+)", os.path.basename(f))[0]))
    out = {}
    for f in ck:
        step = int(re.findall(r"(\d+)", os.path.basename(f))[0]); m = build("lena", f, dev); layers = lena_layers(m)
        rho = torch.stack([L.rho.detach().float() for L in layers])
        rec = {}
        for tag, (ds, idx) in sets.items():
            S, P, Dv, TH, ok = [], [], [], [], 0
            for x, y, _ in DataLoader(Subset(ds, idx), 100, num_workers=4):
                for L in layers: L.log_stats = True
                with torch.autocast("cuda", dtype=torch.bfloat16): pr = m(x.to(dev)).argmax(-1).cpu()
                ok += int((pr == y).sum())
                th = torch.stack([L.last["theta"].float() for L in layers], 1)            # (B, M, r, K)
                S.append(torch.stack([L.last["s"].float() for L in layers], 1).cpu()); P.append(torch.stack([L.last["p"].float() for L in layers], 1).cpu())
                Dv.append((th - rho.unsqueeze(0)).abs().mean(-1).cpu()); TH.append(th.cpu())
                for L in layers: L.log_stats = False; L.last = None
            S, P, Dv, TH = map(torch.cat, (S, P, Dv, TH))
            rec[tag] = dict(s=S.mean().item(), kept=(P / 16).mean().item(), dev=Dv.mean().item(), spread=TH.std(0).mean().item(), acc=ok / len(idx),
                            s_block=S.mean((0, 2)).numpy(), dev_block=Dv.mean((0, 2)).numpy())
        out[step] = rec; print(step, {t: {k: round(v, 4) for k, v in r.items() if isinstance(v, float)} for t, r in rec.items()}, flush=True)
    torch.save(out, os.path.join(a.run, "dynamics.pt")); print("DONE")


if __name__ == "__main__":
    main()
