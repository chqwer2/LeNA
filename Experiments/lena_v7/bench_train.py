"""Training step of one LeNA layer (forward + backward), eager vs static-shape torch.compile; prints compile time."""
import time
import torch
import torch.nn as nn
import lena_v7
from lena_v7 import LeNALinear

torch._dynamo.config.cache_size_limit = 64
torch.manual_seed(0)
base = nn.Linear(4096, 4096, bias=False).cuda().half()
L = LeNALinear(base, r=32, alpha=64, causal=True, use_checkpoint=False).cuda()
with torch.no_grad():
    L.B.normal_(0, 0.01)
x = torch.randn(16, 256, 4096, device="cuda", dtype=torch.half, requires_grad=True)


def run(n):
    torch.cuda.synchronize(); t = time.time()
    for _ in range(n):
        L.zero_grad(); y = L(x); y.float().sum().backward()
    torch.cuda.synchronize(); return (time.time() - t) / n * 1e3, y, L.A2.grad.clone()


lena_v7.set_compile(False); run(2); te, ye, ge = run(10)
lena_v7.set_compile(True, dynamic=False)
t0 = time.time(); run(1); print(f"compile time {time.time() - t0:.0f}s", flush=True)
tc, yc, gc = run(10)
print(f"train (16,256,4096) eager {te:.1f} ms compiled {tc:.1f} ms x{te / tc:.1f} out diff {(ye - yc).abs().max():.2e} "
      f"grad rel diff {((ge - gc).norm() / ge.norm()).item():.2e}", flush=True)
