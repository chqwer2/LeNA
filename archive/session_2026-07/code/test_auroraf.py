import torch, importlib
m = importlib.import_module("peft.tuners.lena.activations")

x = torch.randn(2, 5, 8)
af = m.AuroRAF(mode="dim")
y = af(x)

aur = m.CompAuroRA(mode="dim")
aur._init(x)
# share the same transform params so we compare g-interpolation only
aur.H.data = af.H.data.clone()
aur.ws.data = af.ws.data.clone()
aur.ky.data = af.ky.data.clone()
T = aur(x)

g = torch.sigmoid(af.p).mean().item()
diff = (y - T).abs().max().item()
print(f"g_init(mean)={g:.5f}  max|auroraf - AuroRA_T|={diff:.5f}  (both ~0 => starts at AuroRA)")
print("forward shape:", tuple(y.shape))

# grad flows to p (fallback learnable)
y.sum().backward()
print("p.grad finite:", bool(af.p.grad is not None and torch.isfinite(af.p.grad).all()))
print("OK")
