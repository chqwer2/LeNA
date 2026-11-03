"""Training step (forward + backward) of one LeNA layer, PyTorch path vs fused Triton path (lena_v7.set_triton).
Run on a GPU: python bench_triton.py"""
import time
import torch
import torch.nn as nn
import lena_v7
from lena_v7 import LeNALinear, LoRALinear


def run(L, x, n):
    torch.cuda.synchronize(); t = time.time()
    for _ in range(n):
        L.zero_grad(); x.grad = None; y = L(x); y.float().sum().backward()
    torch.cuda.synchronize(); return (time.time() - t) / n * 1e3, y.detach(), L.A2.grad.clone(), L.rho.grad.clone()


torch.manual_seed(0)
import lena_triton
base = nn.Linear(4096, 4096, bias=False).cuda().half()
L = LeNALinear(base, r=32, alpha=64, causal=True, use_checkpoint=False).cuda()
with torch.no_grad():
    L.B.normal_(0, 0.01)
x = torch.randn(16, 256, 4096, device="cuda", dtype=torch.half, requires_grad=True)
lena_v7.set_triton(True)
for blk, nw in ((32, 8), (16, 4), (32, 4), (64, 8), (16, 2)):
    lena_triton.BLOCK, lena_triton.NUM_WARPS = blk, nw
    run(L, x, 3); print(f"  BLOCK={blk} warps={nw}: {run(L, x, 20)[0]:.2f} ms", flush=True)
lena_triton.BLOCK, lena_triton.NUM_WARPS = 32, 8
lena_v7.set_triton(False)
for causal in (True, False):
    for ckpt in (False, True):
        base = nn.Linear(4096, 4096, bias=False).cuda().half()
        L = LeNALinear(base, r=32, alpha=64, causal=causal, use_checkpoint=ckpt).cuda()
        with torch.no_grad():
            L.B.normal_(0, 0.01)
        x = torch.randn(16, 256, 4096, device="cuda", dtype=torch.half, requires_grad=True)
        res = {}
        for on in (False, True):
            lena_v7.set_triton(on); run(L, x, 3); res[on] = run(L, x, 20)
        lena_v7.set_triton(False)
        (te, ye, ge, re), (tf, yf, gf, rf) = res[False], res[True]
        print(f"causal={causal} ckpt={ckpt} (16,256,4096) r=32 K=16: torch {te:.2f} ms  triton {tf:.2f} ms  x{te / tf:.1f}"
              f"  out diff {(ye - yf).abs().max().item():.2e}  A2 grad rel {((ge - gf).norm() / ge.norm()).item():.2e}"
              f"  rho grad rel {((re - rf).norm() / re.norm()).item():.2e}", flush=True)
base = nn.Linear(4096, 4096, bias=False).cuda().half()
Lo = LoRALinear(base, r=32, alpha=64).cuda()
x = torch.randn(16, 256, 4096, device="cuda", dtype=torch.half, requires_grad=True)
for _ in range(3):
    Lo(x).float().sum().backward()
torch.cuda.synchronize(); t = time.time()
for _ in range(20):
    Lo.zero_grad(); Lo(x).float().sum().backward()
torch.cuda.synchronize(); print(f"LoRA r=32 same shape: {(time.time() - t) / 20 * 1e3:.2f} ms", flush=True)
