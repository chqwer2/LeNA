import ast
F="peft/tuners/lena/layer.py"
s=open(F).read()

# 1) import os
if "\nimport os\n" not in s:
    s=s.replace("from .activations import make_lena_activation",
                "import os\nfrom .activations import make_lena_activation",1)

# 2) SteerHead class before class LeNALinear
if "class SteerHead" not in s:
    cls='''class SteerHead(nn.Module):
    """Input-steered rank-k correction that ESCAPES col(B).
    delta(z) = sum_j s_j(z) * (U @ softmax(Wr_j z)), s_j(z)=<ws_j,z>, ws init 0
    => starts at EXACT LoRA. The write direction U*softmax(Wr z) depends on the input
    code z, so the update leaves the fixed subspace col(B) -- the axis LoRA/AuroRA/CeRA
    cannot reach (their output stays in col(B) for any bottleneck nonlinearity)."""
    def __init__(self, d_out, r, p=4, k=1):
        super().__init__()
        self.k=int(k)
        self.U=nn.Parameter(torch.randn(d_out,int(p))/(float(d_out)**0.5))
        self.Wr=nn.Parameter(torch.randn(self.k,int(p),int(r))/(float(r)**0.5))
        self.ws=nn.Parameter(torch.zeros(self.k,int(r)))
    def forward(self,z):
        zc=z.to(self.U.dtype)                                   # [...,r]
        out=0.0
        for j in range(self.k):
            rho=torch.softmax(zc@self.Wr[j].t(),dim=-1)         # [...,p]
            sj=(zc*self.ws[j]).sum(-1,keepdim=True)             # [...,1]
            out=out+sj*(rho@self.U.t())                         # [...,d_out]
        return out


'''
    s=s.replace("class LeNALinear(nn.Module):",cls+"class LeNALinear(nn.Module):",1)

# 3) steer ModuleDict in __init__
if "self.steer = nn.ModuleDict()" not in s:
    s=s.replace("        self.gate = nn.ModuleDict()",
                "        self.gate = nn.ModuleDict()\n        self.steer = nn.ModuleDict()",1)

# 4) create steer in adapter setup (after act assignment) -- env gated
anchor="""        self.act[adapter_name] = make_lena_activation(
            kind=cfg.lena_activation,
            mode=cfg.lena_flex_mode,
            **act_kwargs,
        )"""
add='''
        if os.environ.get("LENA_STEER"):
            self.steer[adapter_name] = SteerHead(
                self.out_features, r,
                p=int(os.environ.get("LENA_STEER_P", "4")),
                k=int(os.environ.get("LENA_STEER_K", "1")),
            )'''
if "self.steer[adapter_name] = SteerHead" not in s:
    assert anchor in s, "act anchor not found"
    s=s.replace(anchor, anchor+add, 1)

# 5) forward injection after dz = B(h)
if "self.steer[name](z)" not in s:
    assert "        dz = B(h)\n" in s, "dz anchor not found"
    s=s.replace("        dz = B(h)\n",
                "        dz = B(h)\n        if name in self.steer:\n            dz = dz + self.steer[name](z).to(dz.dtype)\n",1)

open(F,"w").write(s)
ast.parse(open(F).read())
print("STEER PATCH OK; import os:", "\nimport os\n" in s, "| SteerHead:", "class SteerHead" in s)
