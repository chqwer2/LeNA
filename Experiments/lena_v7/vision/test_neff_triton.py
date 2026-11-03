"""GPU test of LeNA-S (n_eff) on the fused Triton path (lena_triton NEFF) against the PyTorch path (stein_neff).
  python test_neff_triton.py <old lena_triton backup .py>
(1) plain LeNA: new kernel vs the pre-change kernel, bit-identical outputs and grads (pooled + causal);
(2) n_eff: layer._codes with Triton vs PyTorch eager, codes / s / p / F / rho_tok / bend, grads wrt u, v, rho, omega,
    pooled (ViT grid with CLS) and causal (left padding), deff_scale 1 (paper setting) and 0.02 (keeps 0 < s < 1 so the
    sig2 * deff gradient path is exercised)."""
import importlib.util, sys
sys.path.insert(0, "/mnt/scratch2/users/hchen/FLoRA/Experiments/lena_v7"); sys.path.insert(0, "/mnt/scratch2/users/hchen/FLoRA/Experiments/lena_v7/vision")
import torch, torch.nn as nn, torch.nn.functional as F
import lena_v7 as LV, lena_triton as T, lena_mri_ext as X

dev = "cuda"
spec = importlib.util.spec_from_file_location("lena_triton_old", sys.argv[1])
OLD = importlib.util.module_from_spec(spec); spec.loader.exec_module(OLD)
FAIL = []


def smooth_grid(B, g, r, cls=True, scale=1.2):
    x = torch.randn(B * r, 1, g, g, device=dev)
    k = torch.ones(1, 1, 5, 5, device=dev) / 25
    x = F.conv2d(F.pad(x, (2, 2, 2, 2), mode="circular"), k)
    x = (x / x.std((2, 3), keepdim=True)).reshape(B, r, g * g).transpose(1, 2) * scale
    if cls:
        x = torch.cat([torch.randn(B, 1, r, device=dev), x], 1)
    return x.contiguous()


def inputs(causal, seed=0):
    torch.manual_seed(seed)
    B, r = 4, 8
    if causal:
        N = 150
        u = (torch.randn(B, N, r, device=dev).cumsum(1) * 0.25).clamp(-4, 4)
        w = torch.ones(B, N, r, device=dev); w[0, :17] = 0; w[2, :40] = 0
    else:
        u = smooth_grid(B, 14, r)
        N = u.shape[1]
        w = torch.ones(B, N, r, device=dev)
    v = torch.tanh(1.3 * u) + 0.4 * torch.sin(2 * u) + 0.3 * torch.randn_like(u)
    return u.contiguous(), v.contiguous(), w


def rel(a, b):
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-30)).item()


# ---- (1) plain LeNA: new vs old kernel, bit-identical ----------------------------------------------------------
for causal in (False, True):
    u, v, w = inputs(causal, 1)
    torch.manual_seed(5)
    rho = (torch.linspace(-3, 3, 16, device=dev).repeat(8, 1) + 0.2 * torch.randn(8, 16, device=dev))
    om = torch.tensor(0.2, device=dev)
    outs = []
    for mod in (OLD, T):
        uu, vv, rr, oo = [t.clone().requires_grad_(True) for t in (u, v, rho, om)]
        c, d = mod.fused_codes(uu, vv, w, rr, oo, -3.0, 3.0, 1.0, causal)
        g = torch.randn(c.shape, device=dev, generator=torch.Generator(dev).manual_seed(9))
        (c * g).sum().backward()
        outs.append([c.detach(), d["s"], d["p"], d["F"], uu.grad, vv.grad, rr.grad, oo.grad])
    same = all(torch.equal(a, b) for a, b in zip(*outs))
    print(f"[plain {'causal' if causal else 'pooled'}] new vs old kernel bit-identical (c,s,p,F,du,dv,drho,domega): {same}")
    if not same:
        FAIL.append(f"plain bit-identity {causal}")

# ---- (2) n_eff: Triton vs PyTorch eager -------------------------------------------------------------------------
calls = {"n": 0}


def make_layer(causal):
    torch.manual_seed(3)
    m = nn.Sequential(nn.Linear(16, 16)).to(dev)
    LV.inject(m, ["0"], adapter="lena", r=8, alpha=8, K=16, causal=causal, mode="lena", use_checkpoint=False)
    L = LV.lena_layers(m)[0]
    with torch.no_grad():
        L.rho.add_(0.2 * torch.randn_like(L.rho)); L.omega.fill_(-0.3)
    X.install(m, neff=True)
    return L


def run(L, u, v, w, tri, g):
    LV._TRITON["on"] = tri
    L.zero_grad(set_to_none=True)
    uu, vv = u.clone().requires_grad_(True), v.clone().requires_grad_(True)
    X.EXT["acc"] = {}
    n0 = calls["n"]
    c, d = L._codes(uu, vv, w)
    took = calls["n"] > n0
    (c * g).sum().backward()
    a = X.EXT["acc"]
    return took, dict(c=c.detach(), s=d["s"], p=d["p"], F=d["F"], rho_tok=d.get("rho_tok", torch.zeros(1, device=dev)), bend=torch.tensor(a["bend"]),
                      du=uu.grad, dv=vv.grad, drho=L.rho.grad.clone(), domega=L.omega.grad.clone())


for causal in (False, True):
    L = make_layer(causal)
    orig_codes = LV._TRITON["neff_codes"]

    def counted(*a, **k):
        calls["n"] += 1
        return orig_codes(*a, **k)
    LV._TRITON["neff_codes"] = counted
    for scale in (1.0, 0.02):
        X.EXT["deff_scale"] = scale
        u, v, w = inputs(causal, 2)
        g = torch.randn(u.shape, device=dev, generator=torch.Generator(dev).manual_seed(7))
        tk_t, rt = run(L, u, v, w, True, g)
        tk_p, rp = run(L, u, v, w, False, g)
        X.EXT["neff"] = False                                    # plain LeNA reference (PyTorch) to show deff acts
        _, r0 = run(L, u, v, w, False, g)
        X.EXT["neff"] = True
        s = rp["s"]
        frac = ((s > 0) & (s < 1)).float().mean().item()
        name = f"neff {'causal' if causal else 'pooled'} deff_scale={scale}"
        print(f"[{name}] triton path taken={tk_t} (pytorch run took it={tk_p}); deff mean "
              f"{X.EXT['deff'].float().mean().item():.2f}; s mean neff {s.mean().item():.4f} vs plain "
              f"{r0['s'].mean().item():.4f}; frac 0<s<1 {frac:.3f}")
        line = []
        for k in ("c", "s", "p", "F", "rho_tok", "bend"):
            line.append(f"{k} maxabs {(rt[k] - rp[k]).abs().max().item():.2e}")
        for k in ("du", "dv", "drho", "domega"):
            line.append(f"{k} rel {rel(rt[k], rp[k]):.2e}")
        print("   " + "; ".join(line))
        bad = (not tk_t or tk_p or (rt["c"] - rp["c"]).abs().max().item() > 1e-3 or
               (rt["s"] - rp["s"]).abs().max().item() > 1e-3 or
               max(rel(rt[k], rp[k]) for k in ("du", "dv", "drho", "domega")) > 1e-3)
        if bad:
            FAIL.append(name)
    LV._TRITON["neff_codes"] = orig_codes

# other patches still fall back to PyTorch
X.EXT["ridge"] = True
print("ridge+neff -> triton allowed:", X.triton_neff_ok())
if X.triton_neff_ok():
    FAIL.append("guard")
X.EXT["ridge"] = False
print("FAILED:", FAIL if FAIL else "none")
print("ALL PASSED" if not FAIL else "SOME FAILED")
sys.exit(1 if FAIL else 0)
