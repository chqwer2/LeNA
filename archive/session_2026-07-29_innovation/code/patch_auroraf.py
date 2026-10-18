import ast, sys

CFG = "peft/tuners/lena/config.py"
ACT = "peft/tuners/lena/activations.py"

# --- edit 1: config Literal ---
s = open(CFG).read()
if '"auroraf"' not in s:
    s2 = s.replace('"aurorag"]', '"aurorag", "auroraf"]', 1)
    assert s2 != s, "config Literal anchor not found"
    open(CFG, "w").write(s2)
    print("config: added auroraf")
else:
    print("config: already has auroraf")

# --- edit 2: insert AuroRAF class before CompAuroRA ---
s = open(ACT).read()
CLS = '''class AuroRAF(nn.Module):
    """AuroRA with provable LoRA fallback via per-dim interpolation gate.
    phi(z)=(1-g)*T(z)+g*z, g=sigmoid(p), p init -6 => g~0 => phi~AuroRA at init;
    p->+inf => g->1 => phi=z exact LoRA fallback. No spurious residual (cf aurorag)."""
    kind = "auroraf"

    def __init__(self, mode="dim", n_knots=8, p_init=-6.0, **kw):
        super().__init__(); self.n_knots=int(n_knots); self.p_init=float(p_init)
        self.H=None; self.ws=None; self.ky=None; self.p=None
        self.register_buffer("kx", torch.linspace(-3,3,self.n_knots))
    def _init(self,x):
        if self.H is not None: return
        C=int(x.shape[-1]); d,dev=x.dtype,x.device
        self.H=nn.Parameter(torch.eye(C,dtype=d,device=dev)+0.01*torch.randn(C,C,dtype=d,device=dev))
        self.ws=nn.Parameter(torch.zeros(C,dtype=d,device=dev))
        self.ky=nn.Parameter(self.kx.to(d).view(1,-1).repeat(C,1).clone())
        self.p=nn.Parameter(torch.full((C,), self.p_init, dtype=d, device=dev))
    def forward(self,x):
        self._init(x)
        fixed=torch.tanh(torch.matmul(torch.tanh(x), self.H.transpose(0,1)))
        kx=self.kx.to(x.dtype); idx=torch.searchsorted(kx, x.clamp(kx[0],kx[-1]).contiguous())
        idx=idx.clamp(1,self.n_knots-1)
        yv=self.ky[torch.arange(x.shape[-1],device=x.device), idx]
        T=fixed+self.ws*yv
        g=torch.sigmoid(self.p)
        return (1.0-g)*T + g*x


'''
if "class AuroRAF" not in s:
    anchor = "class CompAuroRA(nn.Module):"
    assert anchor in s, "CompAuroRA anchor not found"
    s = s.replace(anchor, CLS + anchor, 1)
    print("activations: inserted AuroRAF class")
else:
    print("activations: AuroRAF already present")

# --- edit 3: factory branch (insert before aurorag branch) ---
if 'k == "auroraf"' not in s:
    anchor = '    elif k == "aurorag":'
    assert anchor in s, "factory aurorag anchor not found"
    branch = '    elif k == "auroraf":\n        kwargs.pop("use_gate", None)\n        act = AuroRAF(mode=mode, **kwargs)\n'
    s = s.replace(anchor, branch + anchor, 1)
    print("factory: added auroraf branch")
else:
    print("factory: already has auroraf")
open(ACT, "w").write(s)

# verify syntax
ast.parse(open(ACT).read())
ast.parse(open(CFG).read())
print("SYNTAX OK; DONE")
