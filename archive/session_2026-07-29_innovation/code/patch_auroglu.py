import ast
AF="peft/tuners/lena/activations.py"; CF="peft/tuners/lena/config.py"
s=open(AF).read()
cls='''class AuroGLU(nn.Module):
    """AuroRA transform gated by an input-dependent GLU sigmoid:
    sigma(z) = [tanh(H tanh z) + w_s*spline(z)] * sigmoid(W2 z + b).
    Combines additive (AuroRA) + multiplicative (GLU) nonlinearity in the bottleneck."""
    kind = "auroglu"
    def __init__(self, mode="dim", n_knots=8, **kw):
        super().__init__(); self.n_knots=int(n_knots)
        self.H=None; self.ws=None; self.ky=None; self.W2=None; self.b=None
        self.register_buffer("kx", torch.linspace(-3,3,self.n_knots))
    def _init(self,x):
        if self.H is not None: return
        C=int(x.shape[-1]); d,dev=x.dtype,x.device
        self.H=nn.Parameter(torch.eye(C,dtype=d,device=dev)+0.01*torch.randn(C,C,dtype=d,device=dev))
        self.ws=nn.Parameter(torch.zeros(C,dtype=d,device=dev))
        self.ky=nn.Parameter(self.kx.to(d).view(1,-1).repeat(C,1).clone())
        self.W2=nn.Parameter(torch.zeros(C,C,dtype=d,device=dev))
        self.b =nn.Parameter(torch.full((C,),3.0,dtype=d,device=dev))
    def forward(self,x):
        self._init(x)
        fixed=torch.tanh(torch.matmul(torch.tanh(x), self.H.transpose(0,1)))
        kx=self.kx.to(x.dtype); idx=torch.searchsorted(kx, x.clamp(kx[0],kx[-1]).contiguous()).clamp(1,self.n_knots-1)
        yv=self.ky[torch.arange(x.shape[-1],device=x.device), idx]
        aur=fixed+self.ws*yv
        gate=torch.sigmoid(torch.matmul(x,self.W2.transpose(0,1))+self.b)
        return aur*gate


'''
if "class AuroGLU" not in s:
    a="class CompAuroRA(nn.Module):"; s=s.replace(a, cls+a, 1); print("AuroGLU inserted")
if 'k == "auroglu"' not in s:
    fa='    elif k == "glu":'
    s=s.replace(fa, '    elif k == "auroglu":\n        kwargs.pop("use_gate", None)\n        act = AuroGLU(mode=mode, **kwargs)\n'+fa, 1); print("factory auroglu added")
open(AF,"w").write(s); ast.parse(open(AF).read())
c=open(CF).read()
if '"auroglu"' not in c:
    c=c.replace('"auroraf", "glu"]', '"auroraf", "glu", "auroglu"]',1); open(CF,"w").write(c); print("config auroglu added")
ast.parse(open(CF).read()); print("syntax OK")
