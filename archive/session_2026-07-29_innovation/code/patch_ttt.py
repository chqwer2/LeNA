import ast
F = "peft/tuners/lena/layer.py"
s = open(F).read()

# 1) TTTHead class: faithful ViT^3 (arXiv 2512.01643) simplified-SwiGLU inner module, ported
#    to a PEFT write branch. Per forward, fit inner weights (w1,w2) to reconstruct value V from
#    key K over the CURRENT sequence in ONE analytic (hand-derived, closed-form) GD step, then
#    produce the bottleneck code with the UPDATED weights applied to Q. Escapes NOTHING by itself
#    (code -> col(Bt)); its novelty vs LoRA is a CROSS-TOKEN, TEST-TIME-ADAPTED code (w* depends on
#    the whole sequence) instead of a static per-token B*phi(Ax). Bt init 0 => delta=0 at start
#    (exact base/LoRA), Bt gets gradient immediately (LoRA-style safe init). lr=1 per the paper.
cls = '''class TTTHead(nn.Module):
    """ViT^3-style (arXiv 2512.01643) test-time-training inner module as a PEFT write branch.
    q,k,v = low-rank projections of the layer input x (d_in->r). One analytic inner GD step fits
    a simplified-SwiGLU inner model f_{w1,w2} on (k->v) over the CURRENT sequence (hand-derived
    gradient, no 2nd-order autograd; grad-clip g/(||g||+1); lr=1). The updated weights are applied
    to q to give the bottleneck code o=(q w1*)*silu(q w2*), then up-projected: delta = Bt o.
    o is CROSS-TOKEN & TEST-TIME-ADAPTED (w* depends on all tokens), unlike LoRA's static
    per-token B*phi(Ax). Bt init 0 => starts at exact base/LoRA (safe init)."""
    def __init__(self, d_in, d_out, r, lr=1.0):
        super().__init__()
        self.r = int(r); self.lr = float(lr); self.scale = float(r) ** -0.5
        self.Wq = nn.Parameter(torch.randn(int(r), int(d_in)) * (float(d_in) ** -0.5))
        self.Wk = nn.Parameter(torch.randn(int(r), int(d_in)) * (float(d_in) ** -0.5))
        self.Wv = nn.Parameter(torch.randn(int(r), int(d_in)) * (float(d_in) ** -0.5))
        self.w1 = nn.Parameter(torch.empty(int(r), int(r)))
        self.w2 = nn.Parameter(torch.empty(int(r), int(r)))
        nn.init.trunc_normal_(self.w1, std=0.02); nn.init.trunc_normal_(self.w2, std=0.02)
        self.Bt = nn.Parameter(torch.zeros(int(d_out), int(r)))

    def _inner(self, k, v):
        # k,v: [B,N,r]; w1,w2: [r,r] -> updated [B,r,r]. Hand-derived one-step grad (dot-product loss).
        z1 = torch.matmul(k, self.w1)
        z2 = torch.matmul(k, self.w2)
        sig = torch.sigmoid(z2)
        a = z2 * sig                                   # silu(z2)
        N = k.shape[-2]
        e = -v / float(N) * self.scale                 # dl/dv_hat, v_hat=z1*a
        kt = k.transpose(-2, -1)
        g1 = torch.matmul(kt, e * a)
        g2 = torch.matmul(kt, e * z1 * (sig * (1.0 + z2 * (1.0 - sig))))
        g1 = g1 / (g1.norm(dim=-2, keepdim=True) + 1.0)
        g2 = g2 / (g2.norm(dim=-2, keepdim=True) + 1.0)
        return self.w1 - self.lr * g1, self.w2 - self.lr * g2

    def forward(self, x):
        sq = False
        if x.dim() == 2:
            x = x.unsqueeze(0); sq = True
        xf = x.to(self.Wq.dtype)
        q = torch.matmul(xf, self.Wq.t())
        k = torch.matmul(xf, self.Wk.t())
        v = torch.matmul(xf, self.Wv.t())
        w1s, w2s = self._inner(k, v)                   # [B,r,r]
        o = torch.matmul(q, w1s) * torch.nn.functional.silu(torch.matmul(q, w2s))  # [B,N,r]
        delta = torch.matmul(o, self.Bt.t())           # [B,N,d_out]
        if sq:
            delta = delta.squeeze(0)
        return delta.to(x.dtype)


'''
if "class TTTHead" not in s:
    anchor = "class OutGate(nn.Module):"
    assert anchor in s, "OutGate anchor missing"
    s = s.replace(anchor, cls + anchor, 1)
    print("TTTHead class inserted")

# 2) ModuleDict
if "self.ttt = nn.ModuleDict()" not in s:
    s = s.replace("        self.outgate = nn.ModuleDict()",
                  "        self.outgate = nn.ModuleDict()\n        self.ttt = nn.ModuleDict()", 1)
    print("ttt ModuleDict added")

# 3) env-gated creation, after the outgate creation block
anchor2 = '''        if os.environ.get("LENA_OUTGATE"):
            self.outgate[adapter_name] = OutGate(self.out_features, r)
            for p in self.outgate[adapter_name].parameters():
                p.requires_grad = True'''
add2 = '''
        if os.environ.get("LENA_TTT"):
            self.ttt[adapter_name] = TTTHead(self.in_features, self.out_features, r,
                                             lr=float(os.environ.get("LENA_TTT_LR", "1.0")))
            for p in self.ttt[adapter_name].parameters():
                p.requires_grad = True'''
if "self.ttt[adapter_name] = TTTHead" not in s:
    assert anchor2 in s, "outgate creation anchor missing"
    s = s.replace(anchor2, anchor2 + add2, 1)
    print("ttt creation wired")

# 4) forward injection, after the outgate forward block (uses layer input x, NOT z)
anchor3 = '''        if name in self.outgate:
            og = self.outgate[name]
            if og.Wg.device != z.device:
                og.to(z.device)
            dz = og(z, dz)'''
add3 = '''
        if name in self.ttt:
            tt = self.ttt[name]
            if tt.Wq.device != x.device:
                tt.to(x.device)
            dz = dz + tt(x).to(dz.dtype)'''
if "if name in self.ttt:" not in s:
    assert anchor3 in s, "outgate forward anchor missing"
    s = s.replace(anchor3, anchor3 + add3, 1)
    print("ttt forward injected")

open(F, "w").write(s)
ast.parse(open(F).read())
print("syntax OK")
