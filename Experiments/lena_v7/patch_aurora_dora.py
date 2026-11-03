"""Patch the Kelvin copy: AuroRA adapter in lena_v7.py; 'dora' and 'aurora' adapter choices in camelyon.py / clip_fewshot.py."""
import sys

root = sys.argv[1]


def patch(path, pairs):
    s = open(path).read()
    for a, b in pairs:
        assert a in s, (path, a[:60])
        s = s.replace(a, b, 1)
    open(path, "w").write(s)
    print("patched", path)


AURORA = '''

class ANL(nn.Module):
    """Adaptive non-linear layer of AuroRA (Dong et al., NeurIPS 2025; github.com/ins1stenc3/AuroRA, ANL.py),
    copied: tanh(H tanh(z)) + W_s * sum_i B-spline(z_i), cubic B-splines, grid 5 on [-1, 1]."""

    def __init__(self, in_features, out_features, grid_size=5, spline_order=3, scale_noise=0.1, grid_range=(-1, 1)):
        super().__init__()
        self.in_features, self.out_features, self.spline_order = in_features, out_features, spline_order
        h = (grid_range[1] - grid_range[0]) / grid_size
        self.register_buffer("grid", torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0])
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.empty(out_features, grid_size + spline_order))
        nn.init.xavier_normal_(self.base_weight)
        nn.init.uniform_(self.spline_weight, -scale_noise, scale_noise)

    def b_splines(self, x):
        x = x.unsqueeze(-1); g = self.grid
        bases = ((x >= g[:-1]) & (x < g[1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            d1 = g[k:-1] - g[:-k - 1]; d2 = g[k + 1:] - g[1:-k]
            d1 = torch.where(d1 == 0, torch.ones_like(d1), d1); d2 = torch.where(d2 == 0, torch.ones_like(d2), d2)
            bases = (x - g[:-k - 1]) / d1 * bases[..., :-1] + (g[k + 1:] - x) / d2 * bases[..., 1:]
        return bases

    def forward(self, x):
        shp = x.shape; x = x.reshape(-1, self.in_features)
        base = torch.tanh(F.linear(torch.tanh(x), self.base_weight))
        spl = F.linear(self.b_splines(x).sum(dim=1).to(self.spline_weight.dtype), self.spline_weight)
        return (base + spl).reshape(*shp[:-1], self.out_features)


class AuroRALinear(nn.Module):
    """AuroRA baseline: h = W x + (alpha / r) B ANL(A dropout(x)), evaluated with the same dynamic forward
    (AuroRA-D; the released merge path does not run). A: kaiming_uniform(a=sqrt 5) as in the ViT code, B: zeros."""

    def __init__(self, base: nn.Linear, r=2, alpha=4, dropout=0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        dev = base.weight.device
        self.r, self.scaling = r, alpha / r
        self.A1 = nn.Parameter(torch.empty(r, base.in_features, device=dev))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=dev))
        nn.init.kaiming_uniform_(self.A1, a=math.sqrt(5))
        self.anl = ANL(r, r).to(dev)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        z = F.linear(self.drop(x), self.A1.to(x.dtype))
        return self.base(x) + self.scaling * F.linear(self.anl(z.float()).to(x.dtype), self.B.to(x.dtype))
'''
patch(f"{root}/lena_v7.py", [
    ('ADAPTERS = {"lena": LeNALinear, "lora": LoRALinear}', AURORA.lstrip("\n") + '\n\nADAPTERS = {"lena": LeNALinear, "lora": LoRALinear, "aurora": AuroRALinear}'),
    ('        if isinstance(L, (LeNALinear, LoRALinear)):', '        if isinstance(L, (LeNALinear, LoRALinear, AuroRALinear)):'),
])
patch(f"{root}/vision/camelyon.py", [
    ('choices=["lena", "lora", "lora_dagger", "head"]', 'choices=["lena", "lora", "lora_dagger", "dora", "aurora", "head"]'),
    ('        kind = "lena" if a.adapter == "lena" else "lora"\n        kw = dict(r=r, alpha=a.alpha * r / a.r)                     # same alpha/r scale across ranks',
     '        kind = {"lena": "lena", "aurora": "aurora"}.get(a.adapter, "lora")\n        kw = dict(r=r, alpha=a.alpha * r / a.r)                     # same alpha/r scale across ranks\n        if a.adapter == "dora":\n            kw["dora"] = True'),
])
patch(f"{root}/vision/clip_fewshot.py", [
    ('choices=["lena", "lora", "zeroshot"]', 'choices=["lena", "lora", "dora", "aurora", "zeroshot"]'),
    ('        names = inject(model.clip, ["q_proj", "k_proj", "v_proj"], adapter=a.adapter, **kw)',
     '        if a.adapter == "dora":\n            kw["dora"] = True\n        names = inject(model.clip, ["q_proj", "k_proj", "v_proj"], adapter="lora" if a.adapter == "dora" else a.adapter, **kw)'),
])
