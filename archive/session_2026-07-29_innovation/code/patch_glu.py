import ast
AF = "peft/tuners/lena/activations.py"
CF = "peft/tuners/lena/config.py"

s = open(AF).read()
cls = '''class FlexGLU(nn.Module):
    """GLU-style MULTIPLICATIVE bottleneck: glu(z) = (W1 z) * sigmoid(W2 z + b).
    Multiplicative (bilinear) nonlinearity in the code, vs AuroRA's additive tanh/spline.
    Init W1=I, W2=0, b=+3 => gate~0.95 => glu ~= z (near-LoRA start)."""
    kind = "glu"
    def __init__(self, mode="dim", **kw):
        super().__init__(); self.W1=None; self.W2=None; self.b=None
    def _init(self, x):
        if self.W1 is not None: return
        C=int(x.shape[-1]); d,dev=x.dtype,x.device
        self.W1=nn.Parameter(torch.eye(C,dtype=d,device=dev))
        self.W2=nn.Parameter(torch.zeros(C,C,dtype=d,device=dev))
        self.b =nn.Parameter(torch.full((C,),3.0,dtype=d,device=dev))
    def forward(self, x):
        self._init(x)
        return torch.matmul(x, self.W1.transpose(0,1)) * torch.sigmoid(torch.matmul(x, self.W2.transpose(0,1)) + self.b)


'''
if "class FlexGLU" not in s:
    anchor = "class CompAuroRA(nn.Module):"
    assert anchor in s
    s = s.replace(anchor, cls + anchor, 1)
    print("FlexGLU class inserted")
# factory branch
if 'k == "glu"' not in s:
    fa = '    elif k == "aurora":'
    assert fa in s
    s = s.replace(fa, '    elif k == "glu":\n        kwargs.pop("use_gate", None)\n        act = FlexGLU(mode=mode, **kwargs)\n' + fa, 1)
    print("factory glu branch added")
open(AF, "w").write(s)
ast.parse(open(AF).read())

c = open(CF).read()
if '"glu"' not in c:
    c = c.replace('"aurorag", "auroraf"]', '"aurorag", "auroraf", "glu"]', 1)
    open(CF, "w").write(c)
    print("config: glu added")
ast.parse(open(CF).read())
print("syntax OK")
