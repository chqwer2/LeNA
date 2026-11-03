"""Cross-site glioma segmentation with SAM ViT-B (SAMed-style: no prompts, adapters on the image encoder's qkv,
mask decoder fine-tuned) under lora / dora / aurora / lena adapters.

2D axial slices in, 3D per-visit Dice out (WT = NCR+ED+ET, TC = NCR+ET, ET). Model selection on a patient-level
hold-out of the training cohorts (training-domain validation); test cohorts are evaluated once with the selected
checkpoint and once at the last step.

  python mri_sam_seg.py --adapter lena --lr 5e-5 --train_cohorts RHUH-GBM --test_cohorts UCSF-ALPTDG --out runs/x
"""
import argparse
import contextlib
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import lena_v7  # noqa: E402
from lena_v7 import inject, lena_layers, lena_override, lena_context, adapter_tensors, trainable_count  # noqa: E402

DATA = "/mnt/scratch2/users/hchen/data/lena_v7/mri"
SAM = "facebook/sam-vit-base"
PAD = 224                                                            # 192 -> 224 = 14x14 tokens, one SAM window
REGIONS = ("WT", "TC", "ET")
PHASE_COHORTS = {"RHUH-GBM"}                                       # cohorts also reported per phase


# ---------------------------------------------------------------------------------------------------------------
# Data
def load_index(root, cohorts, limit=0):
    out = []
    for c in cohorts:
        vs = json.load(open(os.path.join(root, c, "index.json")))["visits"]
        out += vs[:limit] if limit else vs
    return out


def split_patients(visits, frac, seed):
    pats = sorted({(v["cohort"], v["patient"]) for v in visits})
    rng = np.random.RandomState(seed)
    rng.shuffle(pats)
    k = max(1, int(round(frac * len(pats))))
    val = set(pats[:k])
    return [v for v in visits if (v["cohort"], v["patient"]) not in val], [v for v in visits if (v["cohort"], v["patient"]) in val]


def regions(seg):
    """uint8 label map (..., H, W) -> float (..., 3, H, W) nested regions WT, TC, ET."""
    seg = torch.as_tensor(seg)
    wt, tc, et = seg > 0, (seg == 1) | (seg == 3), seg == 3
    return torch.stack([wt, tc, et], dim=-3).float()


BG = {"value": -3.0}                                                 # background of the prep (z: -3, minmax: 0)


def pad224(x):
    p = (PAD - x.shape[-1]) // 2
    return F.pad(x, (p, PAD - x.shape[-1] - p, p, PAD - x.shape[-2] - p), value=BG["value"])


class Slices(Dataset):
    """Tumour slices plus a fraction of tumour-free brain slices, read lazily from the per-visit npy stacks."""

    def __init__(self, visits, p_empty, seed, flip):
        rng = np.random.RandomState(seed)
        self.visits, self.flip, self.items = visits, flip, []
        for i, v in enumerate(visits):
            t = set(v["tumor_slices"])
            empty = [z for z in v["brain_slices"] if z not in t]
            k = min(len(empty), int(round(p_empty * len(t))))
            self.items += [(i, z) for z in sorted(t)] + [(i, int(z)) for z in rng.choice(empty, k, replace=False)]

    def __len__(self):
        return len(self.items)

    def _slice(self, i, name, z, dtype):
        """Open the memmap per sample (no cache: ~2 files x 1000+ visits per worker exceeds the open-file limit)."""
        m = np.load(os.path.join(self.visits[i]["path"], f"{name}.npy"), mmap_mode="r")
        out = np.array(m[z], dtype=dtype)
        del m
        return out

    def __getitem__(self, j):
        i, z = self.items[j]
        x = torch.from_numpy(self._slice(i, "img", z, np.float32))
        y = regions(self._slice(i, "seg", z, np.uint8))
        if self.flip and torch.rand(()) < 0.5:
            x, y = x.flip(-1), y.flip(-1)
        return x, y, i


class ScanBatches(torch.utils.data.Sampler):
    """N11: each batch = `scans` random training visits x `per` slices of each (same tumour/empty mix as Slices)."""

    def __init__(self, ds, scans, per, seed):
        self.by, self.scans, self.per = {}, scans, per
        for j, (i, _) in enumerate(ds.items):
            self.by.setdefault(i, []).append(j)
        self.keys, self.rng = sorted(self.by), np.random.RandomState(seed)

    def __len__(self):
        return 10 ** 9

    def __iter__(self):
        while True:
            out = []
            for i in self.rng.choice(self.keys, self.scans, replace=False):
                out += self.rng.choice(self.by[i], self.per, replace=len(self.by[i]) < self.per).tolist()
            yield out


# ---------------------------------------------------------------------------------------------------------------
# Model
class Flat4D(nn.Module):
    """SAM's encoder applies qkv to (B, H, W, C) windows; LeNA expects (B, N, C), one curve per window."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        if x.dim() == 4:
            b, h, w, c = x.shape
            return self.inner(x.reshape(b, h * w, c)).reshape(b, h, w, -1)
        return self.inner(x)


def build_sam(path, res):
    from transformers import SamConfig, SamModel
    cfg = SamConfig.from_pretrained(path)
    g = res // cfg.vision_config.patch_size
    cfg.vision_config.image_size = res
    cfg.prompt_encoder_config.image_size = res
    cfg.prompt_encoder_config.image_embedding_size = g
    sd = SamModel.from_pretrained(path).state_dict()
    pe = sd["vision_encoder.pos_embed"].permute(0, 3, 1, 2)          # absolute pos embed 64x64 -> g x g
    sd["vision_encoder.pos_embed"] = F.interpolate(pe, (g, g), mode="bicubic", align_corners=False).permute(0, 2, 3, 1)
    for i in cfg.vision_config.global_attn_indexes:                  # global-attention relative positions
        for ax in ("h", "w"):
            k = f"vision_encoder.layers.{i}.attn.rel_pos_{ax}"
            sd[k] = F.interpolate(sd[k].t()[None], 2 * g - 1, mode="linear", align_corners=False)[0].t()
    model = SamModel(cfg)
    model.load_state_dict(sd, strict=True)
    return model


class SamSeg(nn.Module):
    def __init__(self, path, res, stat_mask=None):
        super().__init__()
        self.sam, self.res, self.stat_mask = build_sam(path, res), res, stat_mask
        with torch.no_grad():
            self.register_buffer("pe", self.sam.get_image_wide_positional_embeddings(), persistent=False)

    def forward(self, x):                                            # x (B, 3, 192, 192) -> logits (B, 3, 192, 192)
        b = x.shape[0]
        x = pad224(x)
        if self.res != PAD:
            x = F.interpolate(x, (self.res, self.res), mode="bilinear", align_corners=False)
        if self.stat_mask == "brain":                               # LeNA statistics over brain patches only
            g0 = self.res // 16
            fg = (x != BG["value"]).any(1, keepdim=True).float()
            tok = F.adaptive_avg_pool2d(fg, (g0, g0)).flatten(1) >= 0.5          # (B, N) patch mostly brain
            with lena_context(attention_mask=tok):
                emb = self.sam.vision_encoder(x).last_hidden_state
        else:
            emb = self.sam.vision_encoder(x).last_hidden_state
        g = emb.shape[-1]
        sparse = torch.zeros(b, 1, 1, emb.shape[1], device=x.device, dtype=emb.dtype)
        dense = self.sam.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(b, -1, g, g)
        masks = self.sam.mask_decoder(emb, self.pe.expand(b, -1, -1, -1), sparse, dense, multimask_output=True)[0]
        m = F.interpolate(masks[:, 0].float(), (PAD, PAD), mode="bilinear", align_corners=False)
        p = (PAD - 192) // 2
        return m[..., p:p + 192, p:p + 192]


def add_sora_gates(model):
    """SoRA-style gate g in R^r between down and up projection: B_eff = B diag(g) (flag --sora; default off)."""
    gates = []
    for m in model.modules():
        if hasattr(m, "B") and isinstance(m._parameters.get("B"), nn.Parameter) and hasattr(m, "A1"):
            B0 = m._parameters.pop("B")
            m.register_parameter("B_raw", B0)
            m.register_parameter("gate", nn.Parameter(torch.ones(B0.shape[1], device=B0.device)))
            m.register_forward_pre_hook(lambda mod, args: setattr(mod, "B", mod.B_raw * mod.gate[None, :]))
            gates.append(m.gate)
    return gates


@torch.no_grad()
def sora_prox(gates, step_size):
    for g in gates:
        g.copy_(torch.sign(g) * (g.abs() - step_size).clamp_min(0))


@torch.no_grad()
def intensity_shift(x, gen=None):
    """MRI-realistic random shift per slice and channel (built without grad): min-max to [0,1] inside the brain,
    gamma ~ U[0.7, 1.5], smooth multiplicative bias field from a 4x4 control grid (+-20%), re-z-score in the brain."""
    bg = BG["value"]
    brain = (x != bg).any(1, keepdim=True)
    b, c = x.shape[:2]
    big, small = torch.finfo(x.dtype).max, -torch.finfo(x.dtype).max
    lo = torch.where(brain, x, torch.full_like(x, big)).amin((2, 3), keepdim=True)
    hi = torch.where(brain, x, torch.full_like(x, small)).amax((2, 3), keepdim=True)
    u = ((x - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1)
    gamma = torch.empty(b, c, 1, 1, device=x.device).uniform_(0.7, 1.5)
    u = u ** gamma
    field = F.interpolate(torch.empty(b, c, 4, 4, device=x.device).uniform_(0.8, 1.2), x.shape[-2:], mode="bicubic",
                          align_corners=True)
    u = u * field
    m = brain.expand_as(u).float(); n = m.sum((2, 3), keepdim=True).clamp_min(1)
    mu = (u * m).sum((2, 3), keepdim=True) / n
    sd = (((u - mu) ** 2) * m).sum((2, 3), keepdim=True).div(n).sqrt().clamp_min(1e-6)
    z = ((u - mu) / sd).clamp(-5, 5)
    return torch.where(brain.expand_as(z), z, torch.full_like(z, bg))


class EmbedLeNA(nn.Module):
    """R7: one LeNA module on the patch embedding (before the first block's LayerNorm), residual, zero at init."""

    def __init__(self, pe, dim, K, r=8, alpha=16):
        super().__init__()
        from lena_v7 import LeNALinear
        base = nn.Linear(dim, dim)
        nn.init.zeros_(base.weight); nn.init.zeros_(base.bias)
        self.pe, self.L = pe, LeNALinear(base, r=r, alpha=alpha, K=K, mode="lena", causal=False, use_checkpoint=False)

    def forward(self, x):
        h = self.pe(x)
        b, H, W, c = h.shape
        return h + self.L(h.reshape(b, H * W, c)).reshape(b, H, W, c)


def build_adapters(model, a):
    for q in model.sam.parameters():
        q.requires_grad_(False)
    for q in model.sam.mask_decoder.parameters():                    # SAMed: decoder is fine-tuned
        q.requires_grad_(True)
    kind = {"lena": "lena", "aurora": "aurora"}.get(a.adapter, "lora")
    kw = dict(r=a.r, alpha=a.alpha)
    if a.adapter == "dora":
        kw["dora"] = True
    if kind == "lena":
        kw.update(K=getattr(a, "knots", 16), mode=a.mode, causal=False, use_checkpoint=False)
    enc = model.sam.vision_encoder
    lb = getattr(a, "lena_blocks", None)
    if kind == "lena" and lb:                                        # R2: LeNA in blocks lo-hi, LoRA elsewhere
        lo, hi = map(int, lb.split("-"))
        lt = [n for n, m in enc.named_modules() if isinstance(m, nn.Linear) and any(n.endswith(t) for t in a.targets)
              and lo <= int(n.split(".")[1]) <= hi]
        names = inject(enc, lt, adapter="lena", **kw)
        names += inject(enc, a.targets, adapter="lora", r=a.r, alpha=a.alpha)   # remaining nn.Linear targets only
    else:
        if kind == "lena" and getattr(a, "ttr", False):              # R12: cross-fit + query + CV shrinkage layer
            import lena_ttr
            lena_ttr.register()
            kind = "lena_ttr"
        if kind == "lena" and getattr(a, "qt", False):               # R3: quantile transport layer
            import lena_qt
            lena_qt.register()
            kind = "lena_qt"
        names = inject(enc, a.targets, adapter=kind, **kw)
    for L in lena_layers(model):
        L.gate_drop = a.gate_drop
    for n in names:
        parent, child = enc.get_submodule(n.rsplit(".", 1)[0]), n.rsplit(".", 1)[1]
        setattr(parent, child, Flat4D(getattr(parent, child)))
    return names


# ---------------------------------------------------------------------------------------------------------------
# Loss / evaluation
def seg_loss(logits, y):
    bce = F.binary_cross_entropy_with_logits(logits, y)
    p = torch.sigmoid(logits)
    inter, den = (p * y).sum((0, 2, 3)), (p + y).sum((0, 2, 3))
    return bce + (1 - (2 * inter + 1) / (den + 1)).mean()


def dice3d(pred, gt):
    """BraTS convention: both empty -> 1, one empty -> 0."""
    s = pred.sum() + gt.sum()
    return 1.0 if s == 0 else float(2 * (pred & gt).sum() / s)


POOL = {"on": False}                                                # N11: whole volume = one batch, one pooled fit


def scan_ctx(n, dev):
    import lena_mri_ext
    return lena_mri_ext.groups(torch.zeros(n, dtype=torch.long, device=dev)) if POOL["on"] else contextlib.nullcontext()


@torch.no_grad()
def eval_visits(model, visits, dev, bs=40):
    model.eval()
    per, t_inf = [], 0.0
    for v in visits:
        img = np.load(os.path.join(v["path"], "img.npy"), mmap_mode="r")
        gt = regions(np.load(os.path.join(v["path"], "seg.npy"))).bool()          # (Z, 3, H, W)
        torch.cuda.synchronize(); t1 = time.time()
        out, bsv = [], img.shape[0] if POOL["on"] else bs
        for z0 in range(0, img.shape[0], bsv):                       # every axial slice of the volume
            x = torch.from_numpy(np.array(img[z0:z0 + bsv], dtype=np.float32)).to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16), scan_ctx(x.shape[0], dev):
                out.append((model(x) > 0).cpu())
        torch.cuda.synchronize(); t_inf += time.time() - t1
        pred = torch.cat(out)
        per.append(dict(cohort=v["cohort"], phase=v.get("phase", ""), patient=v["patient"], visit=v["visit"],
                        **{r: dice3d(pred[:, k], gt[:, k]) for k, r in enumerate(REGIONS)},
                        **{f"gt_{r}": int(gt[:, k].sum()) for k, r in enumerate(REGIONS)}))
        per[-1]["mean"] = float(np.mean([per[-1][r] for r in REGIONS]))
    res, groups = {}, {}
    for p_ in per:
        groups.setdefault(p_["cohort"], []).append(p_)
    for c in sorted({p_["cohort"] for p_ in per}):
        phases = sorted({p_["phase"] for p_ in per if p_["cohort"] == c})
        if len(phases) > 1 and c in PHASE_COHORTS:
            for ph in phases:
                groups[f"{c}:{ph}"] = [p_ for p_ in per if p_["cohort"] == c and p_["phase"] == ph]
    for c, rows in sorted(groups.items()):
        res[c] = {r: float(np.mean([p[r] for p in rows])) for r in REGIONS}
        res[c]["mean"] = float(np.mean([res[c][r] for r in REGIONS])); res[c]["n"] = len(rows)
    model.train()
    return res, per, t_inf / max(1, len(visits))


def pooled(res):
    top = {k: v for k, v in res.items() if ":" not in k}
    n = sum(v["n"] for v in top.values())
    return sum(v["mean"] * v["n"] for v in top.values()) / max(1, n)


@torch.no_grad()
def param_move(model, snap):
    """Relative movement of each adapter tensor type from its value at the start of training, averaged over modules:
    |x - x0| / |x0| (rho, A1, A2, kappa), norm ratio |x| / |x0| (A2, A1), |B| (B starts at 0), omega values."""
    out = {}
    for n_, q in model.named_parameters():
        leaf = n_.rsplit(".", 1)[-1]
        if n_ not in snap:
            continue
        x, x0 = q.detach().float(), snap[n_].float()
        d = out.setdefault(leaf, {"rel_move": [], "norm_ratio": [], "norm": []})
        d["norm"].append(x.norm().item())
        if x0.norm() > 0:
            d["rel_move"].append(((x - x0).norm() / x0.norm()).item())
            d["norm_ratio"].append((x.norm() / x0.norm()).item())
    return {k: {kk: (float(np.mean(v)) if v else None) for kk, v in d.items()} | {"n": len(d["norm"])} for k, d in out.items()}


def write_rows(path, rows):
    keys = list(rows[0])
    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")


@torch.no_grad()
def curve_stats(model, visits, dev, bs=40):
    """Per-visit mean of each LeNA layer's fitted curve (theta - prior rho, shape r x K) and of the scalar step
    diagnostics, over the brain slices of the visit. For showing that fitted curves separate by site."""
    model.eval()
    Ls = lena_layers(model)
    for L in Ls:
        L.log_stats = True
    dev_curves, scal, meta = [], {}, []
    for v in visits:
        img = np.load(os.path.join(v["path"], "img.npy"), mmap_mode="r")
        zs = np.array(v["brain_slices"])
        acc_t, acc_s, n = [0.0] * len(Ls), [dict() for _ in Ls], 0
        bsv = len(zs) if POOL["on"] else bs
        for z0 in range(0, len(zs), bsv):
            x = torch.from_numpy(np.array(img[zs[z0:z0 + bsv]], dtype=np.float32)).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16), scan_ctx(x.shape[0], dev):
                model(x)
            b = x.shape[0]
            for li, L in enumerate(Ls):
                d = L.last
                th = (d["theta"].float() - L.rho.float().reshape(1, *d["theta"].shape[1:])) if hasattr(L, "rho") and L.rho.numel() == d["theta"][0].numel() else d["theta"].float()
                acc_t[li] = acc_t[li] + th.reshape(-1, b, *th.shape[1:]).mean(0).sum(0).cpu()
                acc_s[li]["bend"] = acc_s[li].get("bend", 0.0) + th.abs().amax(-1).mean(-1).reshape(-1, b).mean(0).sum().item()
                for k, t in d.items():
                    if k in ("theta", "u", "v", "bend") or not torch.is_tensor(t) or t.dim() == 0:
                        continue
                    acc_s[li][k] = acc_s[li].get(k, 0.0) + t.float().reshape(t.shape[0], -1).mean(1).reshape(-1, b).mean(0).sum().item()
            n += b
        dev_curves.append(torch.stack([t / n for t in acc_t]).numpy())
        for li in range(len(Ls)):
            for k, t in acc_s[li].items():
                scal.setdefault(k, np.zeros((len(visits), len(Ls))))[len(meta), li] = t / n
        meta.append([v["cohort"], v.get("phase", ""), v["patient"], v["visit"]])
    for L in Ls:
        L.log_stats = False
    model.train()
    return dict(theta_dev=np.stack(dev_curves), meta=np.array(meta), **{f"stat_{k}": a for k, a in scal.items()})


# ---------------------------------------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=DATA)
    p.add_argument("--sam", default=SAM)
    p.add_argument("--bg", type=float, default=-3.0, help="padding value; must match the prep background")
    p.add_argument("--train_cohorts", nargs="+", required=True)
    p.add_argument("--test_cohorts", nargs="+", required=True)
    p.add_argument("--adapter", choices=["lena", "lora", "dora", "aurora"], default="lena")
    p.add_argument("--targets", nargs="+", default=["qkv"])
    p.add_argument("--mode", default="lena")
    p.add_argument("--gate_drop", type=float, default=0.0, help="LeNA: per-input probability of training with the shared curve only (s=0)")
    p.add_argument("--r", type=int, default=8)
    p.add_argument("--alpha", type=float, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--res", type=int, default=224, choices=[224, 448])
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--p_empty", type=float, default=0.25)
    p.add_argument("--holdout", type=float, default=0.1)
    p.add_argument("--test_holdout", type=float, default=0.2, help="patients held out when a test cohort is trained on")
    p.add_argument("--limit", type=int, default=0, help="smoke tests: visits per cohort")
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--no_overrides", action="store_true")
    p.add_argument("--no_curves", action="store_true")
    # MRI fit sweep (all default off)
    p.add_argument("--stat_mask", choices=["none", "brain"], default="none", help="LeNA stats over brain patches only")
    p.add_argument("--omega_lr", type=float, default=0.0, help="absolute lr for omega (overrides --omega_lr_mult: mult = omega_lr / lr)")
    p.add_argument("--omega_lr_mult", type=float, default=1.0, help="H18: lr factor for the per-layer Stein scale omega only (no decay); omega leaves the curve group")
    p.add_argument("--eval_overrides", default=None, help="eval-only: best.pt of a finished run (same adapter args); per-visit Dice of the learned/fixed/plain passes -> per_visit_<mode>.csv in --out, then exit")
    p.add_argument("--init_from", default=None, help="warm start: load adapter A1/B + mask decoder from a LoRA best.pt")
    p.add_argument("--curve_lr_mult", type=float, default=1.0, help="lr multiplier for LeNA rho/omega/A2")
    p.add_argument("--B_lr_mult", type=float, default=1.0, help="lr multiplier for B (LoRA+)")
    p.add_argument("--sora", type=float, default=0.0, help="SoRA gate with proximal L1, lambda (0 = off)")
    p.add_argument("--aux_shift", action="store_true", help="loss = 0.5 main + 0.5 on an intensity-shifted copy")
    p.add_argument("--ridge", action="store_true", help="N10: per-knot ridge toward the prior instead of the Stein step")
    p.add_argument("--kappa0", type=float, default=4.0)
    p.add_argument("--smooth", action="store_true", help="N14: second-difference smoothness prior fit (s = 1)")
    p.add_argument("--pool_scan", action="store_true", help="N11: LeNA fit pooled over the slices of one scan")
    p.add_argument("--scan_batches", action="store_true", help="batches of 2 scans x bs/2 slices (implied by --pool_scan)")
    p.add_argument("--grad_clip", type=float, default=0.0, help="clip total grad norm (0 = off)")
    # LeNA redesign R1-R7 (default off)
    p.add_argument("--crossfit", action="store_true", help="R5: fit on checkerboard half A, apply to B and vice versa")
    p.add_argument("--neff", action="store_true", help="R1: inflate the Stein noise variance by 1+(N-1)rho_tok")
    p.add_argument("--knots", type=int, default=16, help="R6: number of spline knots K")
    p.add_argument("--monotone", action="store_true", help="R6: monotone rho and isotonic projection of the fit")
    p.add_argument("--lena_at_embed", action="store_true", help="R7: one LeNA module on the patch embedding")
    p.add_argument("--lena_blocks", default=None, help="R2: e.g. 0-3 = LeNA in these blocks, LoRA in the rest")
    p.add_argument("--ttr", action="store_true", help="R12: lena_ttr layer (cross-fit, separate query, CV shrinkage)")
    p.add_argument("--qt", action="store_true", help="R3: lena_qt quantile-transport layer")
    p.add_argument("--pivot_floor", type=float, default=0.0, help="H11: add lambda_min*tr(n)/K to the fit diagonal (0 = off)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    if a.omega_lr > 0:
        a.omega_lr_mult = a.omega_lr / a.lr
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    BG["value"] = a.bg
    os.makedirs(a.out, exist_ok=True)
    dev, t0 = "cuda", time.time()

    tr_all = load_index(a.data, a.train_cohorts, a.limit)
    te = load_index(a.data, [c for c in a.test_cohorts if c not in a.train_cohorts], a.limit)
    if any(c in a.train_cohorts for c in a.test_cohorts):             # in-domain test: disjoint patients
        tr_all, te_in = split_patients(tr_all, a.test_holdout, a.seed + 1)
        te += [v for v in te_in if v["cohort"] in a.test_cohorts]
    tr, va = split_patients(tr_all, a.holdout, a.seed)
    ds = Slices(tr, a.p_empty, a.seed, flip=True)
    g = torch.Generator(); g.manual_seed(a.seed)
    if a.pool_scan or a.scan_batches:
        dl = DataLoader(ds, batch_sampler=ScanBatches(ds, 2, a.bs // 2, a.seed), num_workers=a.workers,
                        pin_memory=True, persistent_workers=a.workers > 0)
    else:
        dl = DataLoader(ds, a.bs, sampler=RandomSampler(ds, replacement=True, num_samples=10 ** 9, generator=g),
                        num_workers=a.workers, pin_memory=True, drop_last=True, persistent_workers=a.workers > 0)

    model = SamSeg(a.sam, a.res, stat_mask=a.stat_mask if a.stat_mask != "none" else None)
    adapted = build_adapters(model, a)
    gates = add_sora_gates(model) if a.sora > 0 else []
    if a.lena_at_embed:
        enc = model.sam.vision_encoder
        enc.patch_embed = EmbedLeNA(enc.patch_embed, enc.config.hidden_size, a.knots)
    if a.ridge or a.pool_scan or a.smooth or a.crossfit or a.neff or a.monotone or a.pivot_floor > 0:
        import lena_mri_ext
        lena_mri_ext.install(model, ridge=a.ridge, kappa0=a.kappa0, smooth=a.smooth, crossfit=a.crossfit,
                             neff=a.neff, monotone=a.monotone, pivot_floor=a.pivot_floor)
        POOL["on"] = a.pool_scan
    model.to(dev)
    if a.init_from:
        sd0 = torch.load(a.init_from)
        own = dict(model.named_parameters())
        take = {k: v for k, v in sd0.items() if k in own and own[k].shape == v.shape and (k.endswith((".A1", ".B")) or ".mask_decoder." in k)}
        with torch.no_grad():
            for k, v in take.items():
                own[k].copy_(v.to(dev))
        print("INIT_FROM", a.init_from, len(take), "tensors", flush=True)
    n_tr = trainable_count(model)
    info = dict(vars(a), adapted=len(adapted), trainable=n_tr, n_train_visits=len(tr), n_val_visits=len(va),
                n_test_visits=len(te), n_train_slices=len(ds))
    print(json.dumps(info), flush=True)
    if a.eval_overrides:
        sd = torch.load(a.eval_overrides, map_location="cpu")
        missing = [k for k in sd if k not in model.state_dict()]
        assert not missing, f"checkpoint keys do not match the model: {missing[:3]}"
        model.load_state_dict({k: v.to(dev) for k, v in sd.items()}, strict=False)
        os.makedirs(a.out, exist_ok=True)
        summ = {}
        for c, ov in (("learned", {}), ("fixed", dict(mode="fixed")), ("plain", dict(mode="plain"))):
            with lena_override(**ov):
                summ[c], per_c, _ = eval_visits(model, te, dev)
            write_rows(os.path.join(a.out, f"per_visit_{c}.csv"), [dict(adapter=a.adapter, mode=c, **r) for r in per_c])
            print("EVAL_OVERRIDE", c, json.dumps(summ[c]), flush=True)
        json.dump(dict(ckpt=a.eval_overrides, overrides=summ), open(os.path.join(a.out, "eval_overrides.json"), "w"), indent=1)
        return

    groups = {"base": [], "curve": [], "B": [], "gate": [], "omega": []}
    for n_, q in model.named_parameters():
        if not q.requires_grad:
            continue
        leaf = n_.rsplit(".", 1)[-1]
        k = "omega" if (leaf == "omega" and a.omega_lr_mult != 1.0) else "gate" if leaf == "gate" else "curve" if leaf in ("rho", "omega", "A2", "kappa_raw", "rho_first", "rho_delta") else "B" if leaf in ("B", "B_raw") else "base"
        groups[k].append(q)
    mult = dict(base=1.0, curve=a.curve_lr_mult, B=a.B_lr_mult, gate=1.0, omega=a.omega_lr_mult)
    opt = torch.optim.AdamW([dict(params=v, lr=a.lr * mult[k], weight_decay=0.0 if k in ("gate", "omega") else a.wd, name=k)
                             for k, v in groups.items() if v], lr=a.lr, weight_decay=a.wd)
    print("PARAM_GROUPS", {k: sum(q.numel() for q in v) for k, v in groups.items()}, flush=True)
    init_snap = {n_: q.detach().clone() for n_, q in model.named_parameters()
                 if n_.rsplit(".", 1)[-1] in ("rho", "A2", "omega", "kappa_raw", "B", "A1", "B_raw", "gate")}
    gnorms = []
    total, warm = a.steps, max(1, int(0.05 * a.steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))

    def save(path):
        torch.save(dict(adapter_tensors(model), **{f"sam.mask_decoder.{k}": v.cpu()
                                                   for k, v in model.sam.mask_decoder.state_dict().items()}), path)

    hist, best = [], (-1.0, None)
    step, run, t_train, n_timed = 0, 0.0, 0.0, 0
    torch.cuda.reset_peak_memory_stats()
    model.train()
    it = iter(dl)
    while step < total:
        x, y, vi = next(it)
        torch.cuda.synchronize(); t1 = time.time()
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        if POOL["on"]:
            import lena_mri_ext
            gctx = lambda: lena_mri_ext.groups(torch.unique(vi, return_inverse=True)[1].to(dev))
        else:
            gctx = contextlib.nullcontext
        with torch.autocast("cuda", dtype=torch.bfloat16), gctx():
            logits = model(x)
        loss = seg_loss(logits.float(), y)
        if a.aux_shift:
            xs = intensity_shift(x)
            with torch.autocast("cuda", dtype=torch.bfloat16), gctx():
                loss = 0.5 * loss + 0.5 * seg_loss(model(xs).float(), y)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        opt.zero_grad(set_to_none=True); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_([q for g_ in opt.param_groups for q in g_["params"]],
                                            a.grad_clip if a.grad_clip > 0 else float("inf"))
        gnorms.append(float(gn))
        opt.step()
        if gates:                                                    # proximal step of the L1 penalty
            sora_prox(gates, a.sora * next(g["lr"] for g in opt.param_groups if g.get("name") == "gate"))
        sched.step()
        torch.cuda.synchronize()
        step += 1; run += loss.item()
        if step > 10:                                                # skip warm-up steps in the timing
            t_train += time.time() - t1; n_timed += 1
        if step % 50 == 0:
            print(f"step {step}/{total} loss {run / 50:.4f} {t_train / max(1, n_timed):.3f}s/step "
                  f"gnorm {np.mean(gnorms[-50:]):.3f} max {np.max(gnorms[-50:]):.3f}", flush=True); run = 0.0
        if step % a.eval_every == 0 or step == total:
            vr, _, _ = eval_visits(model, va, dev)
            h = dict(step=step, id_val=pooled(vr), id_val_detail=vr, wall_s=time.time() - t0)
            print(json.dumps(h), flush=True); hist.append(h)
            if h["id_val"] > best[0]:
                best = (h["id_val"], len(hist) - 1); save(os.path.join(a.out, "best.pt"))
    peak_train = torch.cuda.max_memory_allocated() / 2 ** 30

    test_last, _, _ = eval_visits(model, te, dev)
    sd = torch.load(os.path.join(a.out, "best.pt"))
    model.load_state_dict({k: v.to(dev) for k, v in sd.items()}, strict=False)
    test, per, s_vol = eval_visits(model, te, dev)
    sel = hist[best[1]]
    res = dict(info=info, history=hist, selected_step=sel["step"], id_val=sel["id_val"], test=test, test_last=test_last,
               per_visit=per, s_per_step=t_train / max(1, n_timed), infer_s_per_vol=s_vol,
               peak_train_gb=peak_train, wall_s=time.time() - t0)
    if (a.adapter == "lena" or a.lena_at_embed) and not a.no_overrides:                   # element checks on the selected checkpoint
        res["overrides"] = {}
        for c, ov in (("learned", {}), ("fixed", dict(mode="fixed")), ("plain", dict(mode="plain"))):
            with lena_override(**ov):
                res["overrides"][c] = eval_visits(model, te, dev)[0]
            print("OVERRIDE", c, json.dumps(res["overrides"][c]), flush=True)
    write_rows(os.path.join(a.out, "per_visit.csv"), [dict(adapter=a.adapter, **r) for r in per])
    curves = None
    if (a.adapter == "lena" or a.lena_at_embed) and not a.no_curves:
        curves = curve_stats(model, te, dev)
        np.savez_compressed(os.path.join(a.out, "curves.npz"), **curves)
        print("CURVES saved", flush=True)
    if gates:
        res["sora_gates"] = [g.detach().cpu().tolist() for g in gates]
    res["param_move"] = param_move(model, init_snap)                 # on the selected checkpoint
    res["grad_norm"] = dict(mean=float(np.mean(gnorms)), p99=float(np.percentile(gnorms, 99)), max=float(np.max(gnorms)))
    if "curves" in locals() and curves is not None and "stat_s" in curves:
        res["mean_s_test"] = float(curves["stat_s"].mean())
        res["mean_bend_test"] = float(curves["stat_bend"].mean())          # mean over slices/channels of max-knot |theta - rho|
    json.dump(res, open(os.path.join(a.out, "result.json"), "w"), indent=1)
    row = dict(adapter=a.adapter, lr=a.lr, res=a.res, seed=a.seed, steps=a.steps, selected_step=sel["step"],
               id_val=sel["id_val"], test_mean=pooled(test), test_last=pooled(test_last),
               **{f"test_{r}": float(np.mean([p[r] for p in per])) for r in REGIONS},
               s_per_step=res["s_per_step"], infer_s_per_vol=s_vol, peak_train_gb=peak_train, wall_s=res["wall_s"])
    with open(os.path.join(a.out, "summary.csv"), "w") as f:
        f.write(",".join(row) + "\n" + ",".join(str(v) for v in row.values()) + "\n")
    print("RESULT", json.dumps(row))


if __name__ == "__main__":
    main()
