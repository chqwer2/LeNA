import ast
F = "peft/tuners/lena/layer.py"
s = open(F).read()

# 1) OutGate class (output-side multiplicative gating -> input-dependent write subspace B(x))
cls = '''class OutGate(nn.Module):
    """Output-side multiplicative gate that TRULY escapes col(B):
    delta(x) = (B z) * (1 + tanh(Wg z)) = diag(g(x)) B z = B(x) z, an INPUT-DEPENDENT basis
    (rows of B scaled per input). Unlike bottleneck GLU (stays in col(B)) this makes the write
    subspace itself input-dependent -> per-task subspace routing in multi-task.
    Wg: d_out x r, init 0 => gate=1 => exact LoRA start."""
    def __init__(self, d_out, r):
        super().__init__()
        self.Wg = nn.Parameter(torch.zeros(int(d_out), int(r)))
    def forward(self, z, dz):
        g = 1.0 + torch.tanh(torch.matmul(z.to(self.Wg.dtype), self.Wg.transpose(0, 1)))
        return dz * g.to(dz.dtype)


'''
if "class OutGate" not in s:
    anchor = "class SteerHead(nn.Module):"
    assert anchor in s, "SteerHead anchor missing"
    s = s.replace(anchor, cls + anchor, 1)
    print("OutGate class inserted")

# 2) ModuleDict in __init__
if "self.outgate = nn.ModuleDict()" not in s:
    s = s.replace("        self.steer = nn.ModuleDict()",
                  "        self.steer = nn.ModuleDict()\n        self.outgate = nn.ModuleDict()", 1)
    print("outgate ModuleDict added")

# 3) create in adapter setup (env-gated), right after steer creation block
anchor2 = '''            for p in self.steer[adapter_name].parameters():
                p.requires_grad = True'''
add2 = '''
        if os.environ.get("LENA_OUTGATE"):
            self.outgate[adapter_name] = OutGate(self.out_features, r)
            for p in self.outgate[adapter_name].parameters():
                p.requires_grad = True'''
if "self.outgate[adapter_name] = OutGate" not in s:
    assert anchor2 in s, "steer requires_grad anchor missing"
    s = s.replace(anchor2, anchor2 + add2, 1)
    print("outgate creation wired")

# 4) inject in forward after steer add (dz already includes B(h) + optional steer)
anchor3 = '''        if name in self.steer:
            st = self.steer[name]
            if next(st.parameters()).device != z.device:
                st.to(z.device)
            dz = dz + st(z).to(dz.dtype)'''
add3 = '''
        if name in self.outgate:
            og = self.outgate[name]
            if og.Wg.device != z.device:
                og.to(z.device)
            dz = og(z, dz)'''
if "if name in self.outgate:" not in s:
    assert anchor3 in s, "steer forward anchor missing"
    s = s.replace(anchor3, anchor3 + add3, 1)
    print("outgate forward injected")

open(F, "w").write(s)
ast.parse(open(F).read())
print("syntax OK")
