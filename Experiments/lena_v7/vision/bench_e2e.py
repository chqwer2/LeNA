"""End-to-end cost of one training step and one inference forward on the real task models (R6).

Models and adapter configs are built with the task scripts' own code (camelyon.py / domainbed_clip.py, clip_fewshot.py,
mri_sam_seg.py); inputs are synthetic tensors of the task's shape so data loading is excluded.
  python bench_e2e.py --task vit|clip|mri --method lora|dora|aurora|lena|lena_tri|lenaS|lenaS_tri --out x.jsonl
Train step = forward + loss + backward + optimizer step (task's autocast dtype); infer = no-grad forward in eval mode.
Timed with cuda.synchronize, mean over --iters after --warm warm-up iterations (warm-up includes Triton JIT).
"""
import argparse, json, math, os, sys, time, types
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))
import lena_v7  # noqa: E402
from lena_v7 import inject, trainable_count  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--task", required=True); p.add_argument("--method", required=True)
p.add_argument("--warm", type=int, default=5); p.add_argument("--iters", type=int, default=20)
p.add_argument("--out", required=True)
a = p.parse_args()
tri = a.method.endswith("_tri"); base = a.method.replace("_tri", "")
neff = base == "lenaS"
adapter = "lena" if base in ("lena", "lenaS") else base
lena_v7.set_triton(tri)
torch.manual_seed(0); dev = "cuda"


def neff_install(model):
    import lena_mri_ext
    lena_mri_ext.install(model, neff=True); lena_mri_ext.EXT["acc_eval"] = {}


if a.task == "vit":                                    # Camelyon17 / DomainBed: CLIP ViT-B/16 vision tower, 6 targets
    from camelyon import Classifier, TARGETS
    model = Classifier("openai/clip-vit-base-patch16")
    for q in model.tower.parameters():
        q.requires_grad_(False)
    kind = {"lena": "lena", "aurora": "aurora"}.get(adapter, "lora")
    kw = dict(r=8, alpha=16.0)
    if adapter == "dora":
        kw["dora"] = True
    if kind == "lena":
        kw.update(K=16, mode="lena", causal=False, use_checkpoint=False)
    inject(model.tower, TARGETS, adapter=kind, **kw)
    if neff:
        neff_install(model)
    model.to(dev)
    bs, ibs, dt = 32, 256, torch.bfloat16
    x = torch.randn(bs, 3, 224, 224, device=dev); y = torch.randint(0, 2, (bs,), device=dev)
    xi = torch.randn(ibs, 3, 224, 224, device=dev)
    fwd = lambda xx: model(xx)                                                       # noqa: E731
    loss_fn = lambda: F.cross_entropy(fwd(x), y)                                     # noqa: E731
    infer = lambda: fwd(xi)                                                          # noqa: E731
    scaler = None
elif a.task == "clip":                                 # CLIP few-shot ImageNet: q,k,v of both towers, r2, 1000 prompts
    from clip_fewshot import CLIPClassifier, classnames
    model = CLIPClassifier("openai/clip-vit-base-patch16", classnames())
    for q in model.parameters():
        q.requires_grad_(False)
    kw = dict(r=2, alpha=1.0 * math.sqrt(2), dropout=0.25)
    if adapter == "dora":
        kw["dora"] = True
    if adapter == "lena":
        kw.update(K=16, causal=False, use_checkpoint=False, mode="lena")
    kind = {"dora": "lora"}.get(adapter, adapter)
    inject(model.clip, ["q_proj", "k_proj", "v_proj"], adapter=kind, **kw)
    if neff:
        neff_install(model)
    model.cuda()
    bs, ibs, dt = 32, 256, torch.float16
    x = torch.randn(bs, 3, 224, 224, device=dev); y = torch.randint(0, 1000, (bs,), device=dev)
    xi = torch.randn(ibs, 3, 224, 224, device=dev)
    loss_fn = lambda: F.cross_entropy((100.0 * model.image_features(x) @ model.text_features().t()).float(), y)  # noqa
    tf_cache = {}

    def infer():                                                                     # eval: text features once
        if "t" not in tf_cache:
            tf_cache["t"] = model.text_features().float()
        return model.image_features(xi).float() @ tf_cache["t"].t()
    scaler = torch.amp.GradScaler()
elif a.task == "mri":                                  # BraTS: SAM ViT-B encoder qkv, r8, decoder trained, 224, bs16
    import mri_sam_seg as M
    model = M.SamSeg(M.SAM, 224)
    ns = types.SimpleNamespace(adapter=adapter, r=8, alpha=16.0, mode="lena", targets=["qkv"], gate_drop=0.0,
                               knots=16, lena_blocks=None, ttr=False, qt=False)
    M.build_adapters(model, ns)
    if neff:
        neff_install(model)
    model.to(dev)
    bs, ibs, dt = 16, 16, torch.bfloat16
    x = torch.randn(bs, 3, 192, 192, device=dev); yy = (torch.rand(bs, 3, 192, 192, device=dev) > 0.9).float()
    loss_fn = lambda: M.seg_loss(model(x).float(), yy)                              # noqa: E731
    infer = lambda: model(x)                                                         # noqa: E731
    scaler = None
else:
    raise SystemExit("unknown task")

params = [q for q in model.parameters() if q.requires_grad]
opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.01)


def train_step():
    with torch.autocast("cuda", dtype=dt):
        loss = loss_fn()
    opt.zero_grad(set_to_none=True)
    if scaler is not None:
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    else:
        loss.backward(); opt.step()
    return loss


def timed(fn, n):
    torch.cuda.synchronize(); t = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize(); return (time.time() - t) / n


res = dict(task=a.task, method=a.method, triton=tri, neff=neff, trainable=trainable_count(model),
           gpu=torch.cuda.get_device_name(0), bs_train=bs, bs_infer=ibs)
try:
    model.train()
    t0 = time.time(); timed(train_step, a.warm); res["warm_s"] = time.time() - t0
    torch.cuda.reset_peak_memory_stats()
    res["train_s_per_step"] = timed(train_step, a.iters)
    res["peak_train_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
    model.eval(); opt.zero_grad(set_to_none=True); torch.cuda.empty_cache()

    def inf():
        with torch.no_grad(), torch.autocast("cuda", dtype=dt):
            infer()
    timed(inf, 3)
    torch.cuda.reset_peak_memory_stats()
    res["infer_s_per_batch"] = timed(inf, max(5, a.iters // 2))
    res["peak_infer_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
    res["infer_img_per_s"] = ibs / res["infer_s_per_batch"]
    res["status"] = "ok"
except Exception as e:                                 # record, never invent
    res["status"] = f"error: {type(e).__name__}: {str(e)[:300]}"
print("BENCH", json.dumps(res), flush=True)
with open(a.out, "a") as f:
    f.write(json.dumps(res) + "\n")
