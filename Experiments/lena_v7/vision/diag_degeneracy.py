"""Does LeNA degenerate because the per-input fit is evaluated on the very tokens it was fitted on (self-fit)?
Eval only. For every LeNA module, on a small set of inputs (forward pre-hook captures the module input x; the fit is
recomputed with lena_v7's own functions; theta_check = max |recomputed theta - the layer's own theta|).

  u = A1 x, v = A2 x (same dtype path as the layer), b(u) = hat basis, c = theta^T b(u) with the layer's own fit
  (mode of the layer, Stein step included), c_prior = rho^T b(u).
  R2_* = 1 - sum (v - c)^2 / sum (v - vbar)^2, pooled over inputs x channels, vbar per (input, channel) over the
  tokens being scored.

Non-causal (vision) layers:
  R2_fit_all / R2_prior_all   in-sample fit / prior curve on all tokens
  R2_in_half / R2_out_half    theta fitted on a random half of the tokens (seed 0), scored on that half / the other
  R2_loo                      exact leave-self-out: token t scored with theta fitted on all tokens except t
  R2_prior_out                prior curve on the held-out half
Causal (language) layers, positions t = 0..N-1 of each sequence:
  R2_in_*       inclusive prefix statistics (tokens <= t; what the model uses) applied at t
  R2_ex_*       exclusive prefix statistics (tokens < t) applied at t  (held-out)
  R2_prior_*    prior curve;  s_in_*  mean Stein s of the inclusive fit;  buckets t<16, 16-64, 64-256, >256, all
Both:
  corr_c_v, corr_c_prior   median Pearson over tokens per (input, channel)
  share_perinput  ||B (c - c_prior)|| / ||B c|| per input, median;  gap_c_v_out  ||B (c - v)|| / ||B c||, median
  s_mean   mean Stein s of the model's fit

  python diag_degeneracy.py --runs runs/cam_lena_lr5e-5 runs/clip_lena_s1 --n 32 --out runs/diag_degeneracy_v1.csv
  python diag_degeneracy.py --runs ../math/runs/math_lena --n 32 --device cuda --out runs/diag_degeneracy_math_v1.csv
  python diag_degeneracy.py --runs tiny --n 4 --out /tmp/x.csv          # CPU smoke on a random tiny Llama
"""
import argparse
import csv
import json
import math
import os
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import lena_v7  # noqa: E402
from lena_v7 import (inject, LeNALinear, pooled_stats, prefix_stats, stats_dense, hat_index,  # noqa: E402
                     fit_and_step, spline_eval, _CTX)

DEV = {"dev": "cpu"}
BUCKETS = [("lt16", 0, 16), ("16_64", 16, 64), ("64_256", 64, 256), ("gt256", 256, 10 ** 9), ("all", 0, 10 ** 9)]


def info_of(run):
    return SimpleNamespace(**json.load(open(os.path.join(run, "result.json")))["info"])


def load(model, run):
    f = os.path.join(run, "best.pt") if os.path.exists(os.path.join(run, "best.pt")) else os.path.join(run, "adapter.pt")
    sd = torch.load(f, map_location="cpu")
    bad = [k for k in sd if k not in model.state_dict()]
    assert not bad, f"{run}: keys do not match: {bad[:3]}"
    model.load_state_dict(sd, strict=False)
    return model.float().to(DEV["dev"]).eval()


# ------------------------------------------------------------------------------------------- models + inputs
# each returns (model, list of forward closures, module-name scope)
def camelyon(run, n):
    from camelyon import Classifier, ParquetImages, TARGETS, MEAN, STD
    from torchvision import transforms as T
    i = info_of(run)
    m = Classifier(i.clip)
    inject(m.tower, TARGETS, adapter="lena", r=i.rank, alpha=i.alpha * i.rank / i.r, K=16, mode=i.mode, causal=False,
           use_checkpoint=False)
    load(m, run)
    tf = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BILINEAR), T.ToTensor(), T.Normalize(MEAN, STD)])
    xs = []
    for f, keep in (("validation.pkl", {0, 3, 4}), ("test.pkl", None)):       # half ID patches, half hospital-2 patches
        ds = ParquetImages([os.path.join(i.data, f)], tf, keep=keep, limit=n // 2)
        xs += [ds[k][0] for k in range(len(ds))]
    x = torch.stack(xs).to(DEV["dev"])
    return m, [lambda: m(x)], "vision"


def clip(run, n):
    from clip_fewshot import CLIPClassifier, Images, classnames, MEAN, STD, PKL
    from torchvision import transforms as T
    i = info_of(run)
    m = CLIPClassifier(i.clip, classnames())
    inject(m.clip, ["q_proj", "k_proj", "v_proj"], adapter="lena", r=i.r, alpha=i.alpha * math.sqrt(i.r),
           dropout=i.dropout, K=16, causal=False, use_checkpoint=False, mode=getattr(i, "mode", None) or "lena")
    load(m, run)
    tf = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor(),
                    T.Normalize(MEAN, STD)])
    ds = Images(os.path.join(PKL, "val.pkl"), tf, limit=n, seed=0)
    x = torch.stack([ds[k][0] for k in range(len(ds))]).to(DEV["dev"])
    return m, [lambda: m.image_features(x)], "vision_model"


def mri(run, n):
    import mri_sam_seg as M
    i = info_of(run)
    M.BG["value"] = i.bg
    sm = getattr(i, "stat_mask", "none")
    m = M.SamSeg(i.sam, i.res, stat_mask=sm if sm != "none" else None)
    i.gate_drop = getattr(i, "gate_drop", 0.0)
    M.build_adapters(m, i)
    load(m, run)
    tr_all = M.load_index(i.data, i.train_cohorts)
    if any(c in i.train_cohorts for c in i.test_cohorts):
        tr_all, _ = M.split_patients(tr_all, i.test_holdout, i.seed + 1)
    _, va = M.split_patients(tr_all, i.holdout, i.seed)
    xs, per = [], max(1, n // 4)
    for v in va[:4]:                                                   # 4 ID-validation visits x n/4 tumour slices
        img = np.load(os.path.join(v["path"], "img.npy"), mmap_mode="r")
        ts = np.array(v["tumor_slices"]); ts = ts[np.linspace(0, len(ts) - 1, per).astype(int)]
        xs.append(torch.from_numpy(np.array(img[ts], dtype=np.float32)))
    x = torch.cat(xs)[:n].to(DEV["dev"])
    return m, [lambda: m(x)], "vision_encoder"


def _llm(run, n, texts_fn):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = json.load(open(os.path.join(run, "lena_config.json")))
    cuda = DEV["dev"] != "cpu"
    dtype = (torch.bfloat16 if cfg.get("bf16") else torch.float16) if cuda else torch.float32
    tok = AutoTokenizer.from_pretrained(cfg["base_model"], use_fast=True)
    m = AutoModelForCausalLM.from_pretrained(cfg["base_model"], torch_dtype=dtype).to(DEV["dev"])   # no accelerate needed
    kw = dict(cfg["adapter_kwargs"]); kw.pop("use_checkpoint", None)
    inject(m, cfg["targets"], adapter=cfg["adapter"], **kw)
    sd = torch.load(os.path.join(run, "adapter.pt"), map_location="cpu")
    bad = [k for k in sd if k not in m.state_dict()]
    assert not bad, bad[:3]
    for n_, prm in m.named_parameters():
        if n_ in sd:
            prm.data = sd[n_].to(prm.device, torch.float32)
    m.eval()
    texts = texts_fn(n)
    ids = [tok(t, return_tensors="pt", truncation=True, max_length=512)["input_ids"].to(DEV["dev"]) for t in texts]
    print("sequence lengths", sorted(int(x.shape[1]) for x in ids), flush=True)
    return m, [lambda x=x: m(input_ids=x) for x in ids], "model.layers"     # one unpadded sequence per forward


def math_run(run, n):
    def texts(n):                                                      # GSM8K test prompts + gold answers
        data = [d for d in json.load(open("/mnt/scratch2/users/hchen/data/lena_v7/pissa-dataset/metamath/test.json"))
                if d["type"] == "gsm8k"]
        pick = sorted(random.Random(0).sample(range(len(data)), n))
        return [data[k]["instruction"] + " " + str(data[k]["output"]) for k in pick]
    return _llm(run, n, texts)


def commonsense_run(run, n):
    def generate_prompt(instruction, input=None):                   # commonsense/evaluate_lena.py (DoRA protocol), inlined
        if input:
            return ("Below is an instruction that describes a task, paired with an input that provides further context. "
                    "Write a response that appropriately completes the request.\n\n                ### Instruction:\n"
                    f"                {instruction}\n\n                ### Input:\n                {input}\n\n"
                    "                ### Response:\n                ")
        return ("Below is an instruction that describes a task. Write a response that appropriately completes the request. "
                "\n\n                ### Instruction:\n"
                f"                {instruction}\n\n                ### Response:\n                ")
    root = os.path.join(HERE, "..", "commonsense", "dataset")
    tasks = ["boolq", "piqa", "social_i_qa", "hellaswag", "winogrande", "ARC-Easy", "ARC-Challenge", "openbookqa"]

    def texts(n):                                                      # n/8 per task: prompt + gold response
        out, per = [], max(1, n // len(tasks))
        for t in tasks:
            d = json.load(open(os.path.join(root, t, "test.json")))
            for k in sorted(random.Random(0).sample(range(len(d)), per)):
                out.append(generate_prompt(d[k]["instruction"], d[k].get("input")) + d[k]["output"])
        return out[:n]
    return _llm(run, n, texts)


def tiny(run, n):
    import test_llama_tiny as T
    m = T.build().float().eval()
    g = torch.Generator().manual_seed(0)
    ids = [torch.randint(3, 97, (1, L), generator=g) for L in ([8, 40, 100, 120] * n)[:n]]
    return m, [lambda x=x: m(input_ids=x) for x in ids], "model.layers"


TASK = {"cam": camelyon, "clip": clip, "mri": mri, "math": math_run, "llama3": commonsense_run, "tiny": tiny}


# ------------------------------------------------------------------------------------------- analysis
class Acc:
    """Additive accumulator: R2 numerators/denominators, medians of per-(input, channel) or per-input values, means."""

    def __init__(self):
        self.num, self.den, self.med, self.sm, self.cnt, self.check = {}, {}, {}, {}, {}, 0.0

    def r2(self, k, v, c, w):
        ws = w.sum(1, keepdim=True).clamp_min(1)
        vbar = (v * w).sum(1, keepdim=True) / ws
        self.num[k] = self.num.get(k, 0.0) + float(((v - c) ** 2 * w).sum())
        self.den[k] = self.den.get(k, 0.0) + float(((v - vbar) ** 2 * w).sum())

    def add_med(self, k, t):
        self.med.setdefault(k, []).append(t.flatten().float().cpu())

    def mean(self, k, t, w):
        self.sm[k] = self.sm.get(k, 0.0) + float((t * w).sum()); self.cnt[k] = self.cnt.get(k, 0.0) + float(w.sum())

    def result(self):
        out = {f"R2_{k}": (1 - self.num[k] / self.den[k]) if self.den[k] > 0 else float("nan") for k in self.num}
        out.update({k: float(torch.cat(v).median()) for k, v in self.med.items()})
        out.update({k: (self.sm[k] / self.cnt[k]) if self.cnt[k] > 0 else float("nan") for k in self.sm})
        out["theta_check"] = self.check
        return out


def corr(a, b):
    a = a - a.mean(1, keepdim=True); b = b - b.mean(1, keepdim=True)
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1)).clamp_min(1e-12)


def fit(stats, L):
    diag, off, m, e, nt = stats
    return fit_and_step(diag, off, m, e, nt, L.rho.float(), L.omega.float(), L.tau, _CTX.force_mode or L.mode)


@torch.no_grad()
def analyse(L, x, gen, acc):
    if x.dim() == 2:
        x = x.unsqueeze(0)
    sd = torch.promote_types(x.dtype, torch.float32)                   # same reads as LeNALinear.forward
    u = F.linear(x, L.A1.to(x.dtype)).to(sd).float()
    v = F.linear(x, L.A2.to(x.dtype)).to(sd).float()
    if L.anl is not None:
        v = v - L.anl(u)
    Bsz, N, r = u.shape
    causal = L.causal if _CTX.causal is None else _CTX.causal
    w = torch.ones_like(u)
    if _CTX.mask is not None and tuple(_CTX.mask.shape[:2]) == (Bsz, N):
        w = _CTX.mask.to(u.dtype).unsqueeze(-1).expand(Bsz, N, r).contiguous()
    rho = L.rho.float()
    cp = spline_eval(rho, *hat_index(u, L.lo, L.hi, L.K))
    if causal:
        j, lam, *st = prefix_stats(u, v, w, L.lo, L.hi, L.K)
        theta, d = fit(st, L)                                          # (B, N, r, K): inclusive prefix (the model)
        c = spline_eval(theta, j, lam)
        own = list(stats_dense(j, lam, v, w, L.K)) + [w]
        theta_ex, _ = fit([a - b for a, b in zip(st, own)], L)          # exclusive prefix: tokens < t
        c_ex = spline_eval(theta_ex, j, lam)
        pos = torch.arange(N, device=u.device).view(1, N, 1)
        for name, lo, hi in BUCKETS:
            wb = w * ((pos >= lo) & (pos < hi)).to(w.dtype)
            if wb.sum() == 0:
                continue
            acc.r2(f"in_{name}", v, c, wb); acc.r2(f"ex_{name}", v, c_ex, wb); acc.r2(f"prior_{name}", v, cp, wb)
            acc.mean(f"s_in_{name}", d["s"].float(), wb)
        acc.check = max(acc.check, float((theta - L.last["theta"].float()).abs().max())) if L.last is not None else acc.check
    else:
        j, lam, *st = pooled_stats(u, v, w, L.lo, L.hi, L.K)
        theta, d = fit(st, L)                                          # (B, r, K)
        c = spline_eval(theta.unsqueeze(1), j, lam)
        half = (torch.rand(Bsz, N, 1, generator=gen) < 0.5).to(u).expand(Bsz, N, r) * w
        _, _, *sh = pooled_stats(u, v, half, L.lo, L.hi, L.K)
        ch = spline_eval(fit(sh, L)[0].unsqueeze(1), j, lam)
        own = list(stats_dense(j, lam, v, w, L.K)) + [w]                # exact leave-self-out
        th_loo, _ = fit([a.unsqueeze(1) - b for a, b in zip(st, own)], L)
        c_loo = spline_eval(th_loo, j, lam)
        acc.r2("fit_all", v, c, w); acc.r2("prior_all", v, cp, w)
        acc.r2("in_half", v, ch, half); acc.r2("out_half", v, ch, w - half); acc.r2("prior_out", v, cp, w - half)
        acc.r2("loo", v, c_loo, w)
        acc.mean("s_mean", d["s"].float(), torch.ones_like(d["s"]))
        L._diag_theta = theta
    acc.add_med("corr_c_v", corr(c, v)); acc.add_med("corr_c_prior", corr(c, cp))
    Bm = L.B.float()
    out = lambda t: (t @ Bm.t()).flatten(1).norm(dim=1)                # noqa: E731  per input
    oc = out(c).clamp_min(1e-12)
    acc.add_med("share_perinput", out(c - cp) / oc); acc.add_med("gap_c_v_out", out(c - v) / oc)
    if causal:
        acc.mean("s_mean", d["s"].float(), w)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True, help="run dirs; task from the basename prefix " + "/".join(TASK))
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    DEV["dev"] = a.device
    torch.set_grad_enabled(False)
    rows, t0 = [], time.time()
    for run in a.runs:
        name = os.path.basename(run.rstrip("/"))
        task = next(k for k in TASK if name.startswith(k))
        model, fwds, scope = TASK[task](run, a.n)
        if task in ("math", "llama3", "tiny"):
            lena_v7.install_lena_hooks(model)
        mods = [(n_, L) for n_, L in model.named_modules() if isinstance(L, LeNALinear) and scope in n_]
        gen = torch.Generator().manual_seed(0)
        acc = {n_: Acc() for n_, _ in mods}
        # the layer logs its own theta in L.last during its forward; the pre-hook runs first, so compare after
        pre = [L.register_forward_pre_hook(lambda mod, inp, n_=n_: setattr(mod, "_diag_x", inp[0])) for n_, L in mods]
        def post_hook(mod, inp, out, n_):                              # must return None (else replaces the output)
            analyse(mod, mod._diag_x, gen, acc[n_]); _check(mod, acc[n_])
        post = [L.register_forward_hook(lambda mod, inp, out, n_=n_: post_hook(mod, inp, out, n_)) for n_, L in mods]
        for _, L in mods:
            L.log_stats = True
        for k, f in enumerate(fwds):
            f()
            if k % 8 == 0:
                print(f"  {name}: forward {k + 1}/{len(fwds)} [{time.time() - t0:.0f}s]", flush=True)
        for h in pre + post:
            h.remove()
        res = {}
        for n_, L in mods:
            L.log_stats = False; L.last = None; L._diag_x = None
            res[n_] = acc[n_].result()
            rows.append(dict(run=name, module=n_, r=L.r, causal=bool(L.causal), **res[n_]))
        keys = list(dict.fromkeys(k for n_, _ in mods for k in res[n_]))
        summ = {k: float(np.nanmax([res[n_].get(k, np.nan) for n_, _ in mods]) if k == "theta_check"
                         else np.nanmedian([res[n_].get(k, np.nan) for n_, _ in mods])) for k in keys}
        rows.append(dict(run=name, module="MEDIAN over modules", r=mods[0][1].r, causal=bool(mods[0][1].causal), **summ))
        print(name, len(mods), "modules", json.dumps({k: round(v, 3) for k, v in summ.items()}), f"[{time.time() - t0:.0f}s]",
              flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    keys = list(dict.fromkeys(k for r_ in rows for k in r_))
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, keys); w.writeheader()
        w.writerows([{k: (round(v, 5) if isinstance(v, float) else v) for k, v in r_.items()} for r_ in rows])
    print("DONE", a.out)


def _check(L, acc):
    """Non-causal: compare the recomputed theta with the layer's own logged theta (causal is checked in analyse)."""
    t = getattr(L, "_diag_theta", None)
    if t is not None and L.last is not None and "theta" in L.last:
        acc.check = max(acc.check, float((t - L.last["theta"].float()).abs().max()))
    L._diag_theta = None


if __name__ == "__main__":
    main()
