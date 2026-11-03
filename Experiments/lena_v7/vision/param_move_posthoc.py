"""Post-hoc parameter movement for MRI runs that predate the in-run logging: rebuild the model with the run's seed
(same RNG order as mri_sam_seg.main, so the adapter init is identical), snapshot, load best.pt, compare.

  python param_move_posthoc.py runs/mri_s0_lena runs/mri_diag_lena_lr2e-4 ...
"""
import json
import os
import sys
import types

import numpy as np
import torch

import mri_sam_seg as M


def main(dirs):
    for d in dirs:
        info = json.load(open(os.path.join(d, "result.json")))["info"]
        torch.manual_seed(info["seed"]); np.random.seed(info["seed"])
        a = types.SimpleNamespace(**{**dict(gate_drop=0.0, targets=["qkv"], mode="lena"), **info})
        model = M.SamSeg(M.SAM, info.get("res", 224))
        M.build_adapters(model, a)
        if info.get("ridge"):
            import lena_mri_ext
            lena_mri_ext.install(model, ridge=True, kappa0=info.get("kappa0", 4.0))
        snap = {n: q.detach().clone() for n, q in model.named_parameters()
                if n.rsplit(".", 1)[-1] in ("rho", "A2", "omega", "kappa_raw", "B", "A1")}
        sd = torch.load(os.path.join(d, "best.pt"), map_location="cpu")
        model.load_state_dict(sd, strict=False)
        pm = M.param_move(model, snap)
        json.dump(pm, open(os.path.join(d, "param_move.json"), "w"), indent=1)
        short = {k: {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in x.items()} for k, x in pm.items()}
        print(os.path.basename(d.rstrip("/")), json.dumps(short), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
