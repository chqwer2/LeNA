"""Speed of one LeNA layer, eager vs torch.compile of the fit path, at a decode shape and a training shape; checks the
outputs agree. Run on a GPU: python bench_compile.py"""
import time

import torch
import torch.nn as nn

import lena_v7
from lena_v7 import LeNALinear


def bench(L, x, n=20, train=False):
    for _ in range(3):
        y = L(x); (y.float().sum().backward() if train else None)
    torch.cuda.synchronize(); t = time.time()
    for _ in range(n):
        y = L(x); (y.float().sum().backward() if train else None)
    torch.cuda.synchronize()
    return (time.time() - t) / n * 1e3, y


torch.manual_seed(0)
for name, shape, train in (("decode", (4, 1, 4096), False), ("prefill", (4, 200, 4096), False),
                           ("train", (16, 256, 4096), True)):
    base = nn.Linear(4096, 4096, bias=False).cuda().half()
    L = LeNALinear(base, r=32, alpha=64, causal=True, use_checkpoint=False).cuda()
    with torch.no_grad():
        L.B.normal_(0, 0.01)
    x = torch.randn(*shape, device="cuda", dtype=torch.half, requires_grad=train)
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        lena_v7.set_compile(False); te, ye = bench(L, x, train=train)
        lena_v7.set_compile(True); tc, yc = bench(L, x, train=train)
    err = (ye.float() - yc.float()).abs().max().item()
    print(f"{name:8s} {tuple(shape)}  eager {te:7.2f} ms  compiled {tc:7.2f} ms  x{te / tc:5.1f}  max|diff| {err:.2e}",
          flush=True)
