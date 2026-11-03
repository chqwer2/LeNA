"""Inference-only dump for fig_design_effect (CLIP ViT-B/16 16-shot LeNA-S = runs/clip_V2_R1_omega5e-3_s1).

For n random images of ImageNet val and Sketch: per vision LeNA module (12 blocks x q/k/v) and channel (r=2) the tokens
(u, v) of the fit, and the step s, fit statistic F, retained knots p, applied curve theta and token lag-1 correlation
rho_tok of LeNA-S (R1 design-effect step on). Also rho, omega, lo, hi, tau, K per layer. No training, no weights saved.

  python diag_design_effect_clip.py --ckpt runs/clip_V2_R1_omega5e-3_s1/adapter.pt --out runs/diag_design_effect_clip --n 64
"""
import argparse, math, os, sys, time
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import lena_v7  # noqa: E402
from lena_v7 import inject, LeNALinear  # noqa: E402
import lena_mri_ext  # noqa: E402
from clip_fewshot import CLIPClassifier, Images, classnames, PKL  # noqa: E402
from diag_clip import TF  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/clip_V2_R1_omega5e-3_s1/adapter.pt")
    ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--sets", default="val,sketch"); ap.add_argument("--bs", type=int, default=16)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); t0 = time.time()
    lena_v7.set_triton(False)
    m = CLIPClassifier("openai/clip-vit-base-patch16", classnames())
    for q in m.parameters():
        q.requires_grad_(False)
    inject(m.clip, ["q_proj", "k_proj", "v_proj"], adapter="lena", r=2, alpha=math.sqrt(2), dropout=0.0, K=16,
           causal=False, use_checkpoint=False, mode="lena")
    lena_mri_ext.install(m, neff=True)
    lena_mri_ext.EXT["acc_eval"] = None; lena_mri_ext.EXT["deff_scale"] = 1.0
    sd = torch.load(a.ckpt, map_location="cpu")
    missing = [k for k in sd if k not in m.state_dict()]
    assert not missing, missing[:3]
    m.load_state_dict(sd, strict=False); m = m.cuda().eval()
    vis = [(n, L) for n, L in m.named_modules() if isinstance(L, LeNALinear) and "vision_model" in n]
    res = dict(names=[n for n, _ in vis], ckpt=a.ckpt,
               rho=torch.stack([L.rho.detach().float().cpu() for _, L in vis]),
               omega=torch.stack([L.omega.detach().float().cpu().reshape(-1) for _, L in vis]),
               grid=[(float(L.lo), float(L.hi), int(L.K), float(L.tau)) for _, L in vis])
    for s in a.sets.split(","):
        ds = Images(os.path.join(PKL, f"{s}.pkl"), TF, limit=a.n, seed=0)
        rec = {}
        for x, y in DataLoader(ds, a.bs, num_workers=4):
            for _, L in vis:
                L.log_stats = True
            with torch.autocast("cuda", dtype=torch.float16):
                m.image_features(x.cuda())
            keys = [k for k in vis[0][1].last if k in ("s", "F", "p", "theta", "u", "v", "rho_tok")]
            for k in keys:
                rec.setdefault(k, []).append(torch.stack([L.last[k].float() for _, L in vis], 1).cpu())
            for _, L in vis:
                L.log_stats = False; L.last = None
        res[s] = {k: torch.cat(v) for k, v in rec.items()}
        print(s, {k: tuple(v.shape) for k, v in res[s].items()}, f"{time.time() - t0:.0f}s", flush=True)
        print(s, "mean s", res[s]["s"].mean().item(), flush=True)
    torch.save(res, os.path.join(a.out, "design_effect_clip.pt"))
    print(f"DONE {time.time() - t0:.0f}s peak_mem_GB {torch.cuda.max_memory_allocated() / 1e9:.2f}", flush=True)


if __name__ == "__main__":
    main()
