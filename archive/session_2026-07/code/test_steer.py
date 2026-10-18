import torch, importlib
m = importlib.import_module("peft.tuners.lena.layer")
torch.manual_seed(0)
d_out, r, p, N = 64, 2, 4, 400
sh = m.SteerHead(d_out, r, p=p, k=1)
B = torch.randn(d_out, r)
z = torch.randn(N, r)

def erank(Y):
    Y = Y - Y.mean(0, keepdim=True)
    s = torch.linalg.svdvals(Y)
    return float((s.sum()**2)/(s.pow(2).sum()+1e-9))  # participation-ratio effective rank

lora = z @ B.t()                       # LoRA part, output in col(B), rank<=2
# init: ws=0 => steer=0 => exact LoRA
init_steer = sh(z)
print(f"init steer max-abs = {init_steer.abs().max():.2e}  (should be 0 => exact LoRA start)")

# activate steer
with torch.no_grad():
    sh.ws.normal_(0, 1.0)
full = lora + sh(z)
print(f"eff_rank LoRA-only = {erank(lora):.2f}  (<=r={r})")
print(f"eff_rank LoRA+steer = {erank(full):.2f}  (>r => ESCAPES col(B))")
# grad check
full.sum().backward()
print("grads finite:", all(pp.grad is not None and torch.isfinite(pp.grad).all() for pp in sh.parameters()))
print("OK")
