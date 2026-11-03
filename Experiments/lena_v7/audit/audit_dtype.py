"""Eval-audit (read-only w.r.t. existing code): LeNA eval under the original autocast dtype vs full fp32, same checkpoint,
same subset. Camelyon (bf16 autocast in camelyon.py) and CLIP (fp16 autocast in clip_fewshot.py). Also logs agreement."""
import argparse, json, math, os, sys, contextlib, time, pickle
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
HERE = "/mnt/scratch2/users/hchen/FLoRA/Experiments/lena_v7"
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "vision"))
from lena_v7 import inject, lena_context

def amp(dt):
    return torch.autocast("cuda", dtype=dt) if dt is not None else contextlib.nullcontext()

def run_cam(n, ckpt, lr_tag):
    import camelyon as C
    tf_te = C.T.Compose([C.T.Resize(224, interpolation=C.T.InterpolationMode.BILINEAR), C.T.ToTensor(), C.T.Normalize(C.MEAN, C.STD)])
    ds = C.ParquetImages([os.path.join("/mnt/scratch2/users/hchen/data/lena_v7/camelyon17/pkl", "test.pkl")], tf_te, limit=n)
    m = C.Classifier("openai/clip-vit-base-patch16")
    for q in m.tower.parameters(): q.requires_grad_(False)
    inject(m.tower, C.TARGETS, adapter="lena", r=8, alpha=16, K=16, mode="lena", causal=False, use_checkpoint=False)
    sd = torch.load(ckpt, map_location="cpu"); m.load_state_dict(sd, strict=False); m.cuda().eval()
    dl = DataLoader(ds, 256, num_workers=4)
    out = {}
    for name, dt in (("bf16_autocast", torch.bfloat16), ("fp32", None)):
        P = []
        with torch.no_grad():
            for x, y, c in dl:
                with amp(dt):
                    P.append((m(x.cuda()).argmax(-1).cpu() == y))
        out[name] = torch.cat(P)
    return out

def run_clip(n, ckpt, sets):
    import clip_fewshot as CF
    tf_te = CF.T.Compose([CF.T.Resize(224, interpolation=CF.T.InterpolationMode.BICUBIC), CF.T.CenterCrop(224), CF.T.ToTensor(), CF.T.Normalize(CF.MEAN, CF.STD)])
    m = CF.CLIPClassifier("openai/clip-vit-base-patch16", CF.classnames())
    for q in m.parameters(): q.requires_grad_(False)
    inject(m.clip, ["q_proj", "k_proj", "v_proj"], adapter="lena", r=2, alpha=1.0 * math.sqrt(2), dropout=0.25, K=16, causal=False, use_checkpoint=False, mode="lena")
    m.cuda(); sd = torch.load(ckpt, map_location="cpu"); m.load_state_dict(sd, strict=False); m.eval()
    out = {}
    for s in sets:
        full = CF.Images(os.path.join(CF.PKL, f"{s}.pkl"), tf_te)
        classes = sorted(set(full.ys)); del full
        ds = CF.Images(os.path.join(CF.PKL, f"{s}.pkl"), tf_te, limit=n, seed=0)
        dl = DataLoader(ds, 256, num_workers=4)
        for name, dt in (("fp16_autocast", torch.float16), ("fp32", None)):
            with torch.no_grad(), amp(dt):
                tf = m.text_features().float()
            keep = torch.tensor(classes, device="cuda"); remap = torch.full((1000,), -1, dtype=torch.long, device="cuda"); remap[keep] = torch.arange(len(keep), device="cuda")
            tf = tf[keep]; P = []
            with torch.no_grad():
                for x, y in dl:
                    with amp(dt):
                        lg = m.image_features(x.cuda()).float() @ tf.t()
                    P.append((lg.argmax(-1) == remap[y.cuda()]).cpu())
            out[f"{s}/{name}"] = torch.cat(P)
    return out

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--task"); p.add_argument("--n", type=int); p.add_argument("--ckpt"); p.add_argument("--out")
    a = p.parse_args(); t0 = time.time()
    res = run_cam(a.n, a.ckpt, "") if a.task == "cam" else run_clip(a.n, a.ckpt, ["r", "a"])
    summ = {k: round(100 * v.float().mean().item(), 2) for k, v in res.items()}
    keys = sorted(res); 
    for i in range(0, len(keys), 2):
        A, B = res[keys[i]], res[keys[i + 1]]
        summ[f"agree_{keys[i]}_vs_{keys[i+1]}"] = round(100 * (A == B).float().mean().item(), 2)
    summ["peak_mem_GB"] = round(torch.cuda.max_memory_allocated() / 1e9, 1); summ["sec"] = round(time.time() - t0)
    print("AUDIT_DTYPE", a.task, json.dumps(summ), flush=True)
    json.dump(summ, open(a.out, "w"), indent=1)
