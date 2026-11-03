"""R10 / R11 eval-only: run vision/eval_switch.py with a forward-only change to every LeNA layer's per-input fit
(LoRA model untouched), and record mean Stein s and curve bend of the LeNA model.

  python eval_ext.py --variant irls --K 2  -- <eval_switch args>
  python eval_ext.py --variant stream --beta 0.99 --lam 0.5 -- <eval_switch args>
  python eval_ext.py --variant none -- <eval_switch args>          (K = 0 reference with the same logging)

stream: every DataLoader built by eval_switch iterates a fixed random permutation (seed 0) of its split and resets
the EMA at the start of the split. MRI: the visit lists are shuffled once (seed 0) when loaded; the EMA is NOT reset
between the ID and test passes, and the GLI in-domain hold-out is appended after the shuffled external cohorts
(eval_switch builds the list that way). The EMA is updated once per forward batch with the batch-mean statistics.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_switch as ES  # noqa: E402
import lena_mri_ext as X  # noqa: E402


class StreamLoader(torch.utils.data.DataLoader):
    def __init__(self, ds, *args, **kw):
        kw.pop("shuffle", None)
        super().__init__(ds, *args, sampler=_Perm(len(ds)), **kw)

    def __iter__(self):
        X.EXT["stream_gen"] += 1
        return super().__iter__()


class _Perm(torch.utils.data.Sampler):
    def __init__(self, n):
        g = torch.Generator(); g.manual_seed(0)
        self.idx = torch.randperm(n, generator=g).tolist()

    def __len__(self):
        return len(self.idx)

    def __iter__(self):
        return iter(self.idx)


def main():
    argv = sys.argv[1:]
    cut = argv.index("--") if "--" in argv else len(argv)
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["none", "irls", "stream"], required=True)
    p.add_argument("--K", type=int, default=2)
    p.add_argument("--beta", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.5)
    a = p.parse_args(argv[:cut])
    X.install_global(irls=a.K if a.variant == "irls" else 0, stream=a.beta if a.variant == "stream" else 0.0,
                     stream_lam=a.lam)
    if a.variant == "stream":
        ES.DataLoader = StreamLoader
        import mri_sam_seg as M
        orig_load = M.load_index

        def load_shuffled(root, cohorts, limit=0):                  # MRI: fixed shuffled visit order (seed 0)
            vs = orig_load(root, cohorts, limit)
            return [vs[i] for i in np.random.RandomState(0).permutation(len(vs))]
        M.load_index = load_shuffled
    sys.argv = [sys.argv[0]] + argv[cut + 1:]
    out = argv[argv.index("--out", cut) + 1]
    try:
        ES.main()
    finally:
        acc = X.EXT["acc"] or {}
        n = max(1, acc.get("n", 0))
        json.dump(dict(variant=a.variant, K=a.K, beta=a.beta, lam=a.lam, mean_s=acc.get("s", 0) / n,
                       mean_bend=acc.get("bend", 0) / n, n_fits=acc.get("n", 0)),
                  open(os.path.join(out, "ext_stats.json"), "w"), indent=1)
        print("EXT_STATS", json.dumps(dict(mean_s=acc.get("s", 0) / n, mean_bend=acc.get("bend", 0) / n)), flush=True)


if __name__ == "__main__":
    main()
