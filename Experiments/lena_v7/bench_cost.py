"""Per-layer cost of each adapter on one 4096x4096 LLaMA-sized linear (r=32, batch 16 x 256 tokens, bf16 base):
trainable params, train step (fwd+bwd) time, peak memory, and inference forward latency. Prints a CSV.
  python bench_cost.py > cost.csv
"""
import time
import torch
import torch.nn as nn
import lena_v7
from lena_v7 import LeNALinear, LoRALinear, AuroRALinear

torch.manual_seed(0)
B, N, D, R = 16, 256, 4096, 32


def make(kind):
    base = nn.Linear(D, D, bias=False).cuda().to(torch.bfloat16)
    for p in base.parameters():
        p.requires_grad_(False)
    if kind == "LoRA": m = LoRALinear(base, r=R, alpha=2 * R)
    elif kind == "DoRA": m = LoRALinear(base, r=R, alpha=2 * R, dora=True)
    elif kind == "AuroRA": m = AuroRALinear(base, r=R, alpha=2 * R)
    else: m = LeNALinear(base, r=R, alpha=2 * R, causal=True, use_checkpoint=False)
    m = m.cuda()
    for n_, p in m.named_parameters():
        if p.requires_grad and p.dim() == 2 and "B" in n_.split(".")[-1]:
            with torch.no_grad(): p.normal_(0, 0.01)
    return m


def timeit(fn, n=20):
    for _ in range(3): fn()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.time() - t) / n * 1e3


print("adapter,trainable_params,train_ms,peak_train_MB,infer_ms,peak_infer_MB")
for kind in ("LoRA", "DoRA", "AuroRA", "LeNA (eager)", "LeNA (Triton)"):
    lena_v7.set_triton(kind == "LeNA (Triton)")
    m = make(kind); params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    x = torch.randn(B, N, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    def step():
        m.zero_grad(set_to_none=True); y = m(x); y.float().sum().backward()
    m.train(); torch.cuda.reset_peak_memory_stats(); tr = timeit(step); ptr = torch.cuda.max_memory_allocated() / 2**20
    m.eval(); xi = x.detach()
    def inf():
        with torch.no_grad(): m(xi)
    torch.cuda.reset_peak_memory_stats(); ti = timeit(inf); pin = torch.cuda.max_memory_allocated() / 2**20
    print(f"{kind},{params},{tr:.2f},{ptr:.0f},{ti:.2f},{pin:.0f}", flush=True)
    del m, x; torch.cuda.empty_cache()
