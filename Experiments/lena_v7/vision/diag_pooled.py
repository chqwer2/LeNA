"""Domain-pooled statistics on Camelyon17 (label-free test-time adaptation without gradients): for each hospital, pool
the LeNA sufficient statistics (n, m, e, token count) of N unlabeled patches and fit the curves of every evaluation
patch from the pooled sums. Compared with the per-input fit ("own") and with LoRA on the same 500 evaluation patches
per hospital.
  python diag_pooled.py --lena runs/cam_lena_lr5e-5/best.pt --lora runs/cam_lora_lr1e-4/best.pt --out runs/diag_cam
"""
import argparse, contextlib, json, os, sys
import numpy as np, torch
from torch.utils.data import DataLoader, Subset
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from lena_v7 import lena_layers, lena_override  # noqa
from camelyon import ParquetImages, MEAN, STD  # noqa
from diag_camelyon import build, D  # noqa
from torchvision import transforms as T  # noqa


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(); p.add_argument("--lena", required=True); p.add_argument("--lora", required=True); p.add_argument("--out", required=True)
    p.add_argument("--n_eval", type=int, default=500); p.add_argument("--pools", default="1,8,32,128,512")
    a = p.parse_args(); dev = "cuda"; pools = [int(x) for x in a.pools.split(",")]
    tf = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BILINEAR), T.ToTensor(), T.Normalize(MEAN, STD)])
    mle, mlo = build("lena", a.lena, dev), build("lora", a.lora, dev); layers = lena_layers(mle)

    def acc(model, ds, idx, ctx=contextlib.nullcontext):
        ok = 0
        for x, y, _ in DataLoader(Subset(ds, idx), 100, num_workers=4):
            with ctx(), torch.autocast("cuda", dtype=torch.bfloat16):
                ok += int((model(x.to(dev)).argmax(-1).cpu() == y).sum())
        return ok / len(idx)

    def pooled(ds, idx):
        tot = None
        for x, _, _ in DataLoader(Subset(ds, idx), 100, num_workers=4):
            with lena_override(stats="record"), torch.autocast("cuda", dtype=torch.bfloat16):
                mle(x.to(dev))
            cur = [[t.float().sum(0, keepdim=True) for t in L._rec] for L in layers]
            tot = cur if tot is None else [[u + w for u, w in zip(A, B)] for A, B in zip(tot, cur)]
        return tot

    res = {}
    for split, hs in (("validation", (0, 3, 4, 1)), ("test", (2,))):
        ds = ParquetImages([os.path.join(D, f"{split}.pkl")], tf)
        for h in hs:
            rng = np.random.RandomState(0); idx = list(rng.permutation([i for i, c in enumerate(ds.cs) if c == h]))
            ev, pool = idx[:a.n_eval], idx[a.n_eval:a.n_eval + max(pools)]
            r = dict(lora=acc(mlo, ds, ev), own=acc(mle, ds, ev))
            for n in pools:
                st = pooled(ds, pool[:n])
                for L, s_ in zip(layers, st):
                    L._inj = s_
                r[f"pooled_{n}"] = acc(mle, ds, ev, lambda: lena_override(stats="inject"))
            res[h] = r; print("hospital", h, {k: round(100 * v, 2) for k, v in r.items()}, flush=True)
    json.dump(res, open(os.path.join(a.out, "pooled_stats.json"), "w"), indent=1); print("DONE")


if __name__ == "__main__":
    main()
