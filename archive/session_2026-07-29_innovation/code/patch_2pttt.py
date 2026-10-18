import ast
F = "peft/tuners/lena/layer.py"
s = open(F).read()

cls = '''class TwoPathTTT(nn.Module):
    """ViT^3-grounded two-pathway TTT adapter, made CAUSAL (fixes the future-leakage that broke the
    non-causal port on autoregressive generation). Global = causal SwiGLU inner-TTT: weights (w1,w2)
    fit on the CAUSAL prefix (k_1..k_i -> v) via a hand-derived one-step gradient (cumulative state,
    so position i only sees j<=i), applied to q -> (q w1*)*silu(q w2*). Local = causal depthwise
    conv1d (short-range). Codes concatenated and up-projected by Bt (init 0 => starts at base).
    Trained end-to-end with the task's normal loss (like ViT^3), differentiating through the
    closed-form inner step (no 2nd-order autograd)."""
    def __init__(self, d_in, d_out, r, lr=1.0, ksize=4):
        super().__init__()
        self.r = int(r); self.lr = float(lr); self.scale = float(r) ** -0.5; self.ksize = int(ksize)
        self.Wq = nn.Parameter(torch.randn(int(r), int(d_in)) * (float(d_in) ** -0.5))
        self.Wk = nn.Parameter(torch.randn(int(r), int(d_in)) * (float(d_in) ** -0.5))
        self.Wv = nn.Parameter(torch.randn(int(r), int(d_in)) * (float(d_in) ** -0.5))
        self.w1 = nn.Parameter(torch.empty(int(r), int(r))); nn.init.trunc_normal_(self.w1, std=0.02)
        self.w2 = nn.Parameter(torch.empty(int(r), int(r))); nn.init.trunc_normal_(self.w2, std=0.02)
        self.Wq2 = nn.Parameter(torch.randn(int(r), int(d_in)) * (float(d_in) ** -0.5))
        self.conv = nn.Parameter(torch.zeros(int(r), 1, int(ksize)))   # depthwise causal kernel, init 0
        self.Bt = nn.Parameter(torch.zeros(int(d_out), 2 * int(r)))    # up-proj, init 0 => safe start

    def _causal_swiglu(self, q, k, v):
        z1 = torch.matmul(k, self.w1); z2 = torch.matmul(k, self.w2)
        sig = torch.sigmoid(z2); a = z2 * sig
        e = -v * self.scale
        ea = e * a
        eb = e * z1 * (sig * (1.0 + z2 * (1.0 - sig)))
        # causal cumulative inner-gradient states (position i sees only j<=i):
        S1 = torch.cumsum(torch.einsum('bnr,bns->bnrs', k, ea), dim=1)   # [B,N,r,r]
        S2 = torch.cumsum(torch.einsum('bnr,bns->bnrs', k, eb), dim=1)
        qw1 = torch.matmul(q, self.w1) - self.lr * torch.einsum('bnr,bnrs->bns', q, S1)
        qw2 = torch.matmul(q, self.w2) - self.lr * torch.einsum('bnr,bnrs->bns', q, S2)
        return qw1 * torch.nn.functional.silu(qw2)

    def _causal_conv(self, q2):
        xc = q2.transpose(1, 2)                                   # [B,r,N]
        xc = torch.nn.functional.pad(xc, (self.ksize - 1, 0))    # left pad => causal
        o = torch.nn.functional.conv1d(xc, self.conv.to(xc.dtype), groups=self.r)
        return o.transpose(1, 2)                                  # [B,N,r]

    def forward(self, x):
        sq = False
        if x.dim() == 2:
            x = x.unsqueeze(0); sq = True
        xf = x.to(self.Wq.dtype)
        q = torch.matmul(xf, self.Wq.t()); k = torch.matmul(xf, self.Wk.t()); v = torch.matmul(xf, self.Wv.t())
        o1 = self._causal_swiglu(q, k, v)                         # global [B,N,r]
        o2 = self._causal_conv(torch.matmul(xf, self.Wq2.t()))    # local  [B,N,r]
        o = torch.cat([o1, o2], dim=-1)                           # [B,N,2r]
        delta = torch.matmul(o, self.Bt.t())
        if sq:
            delta = delta.squeeze(0)
        return delta.to(x.dtype)


'''
if "class TwoPathTTT" not in s:
    anchor = "class TTTHead(nn.Module):"
    assert anchor in s
    s = s.replace(anchor, cls + anchor, 1)
    print("TwoPathTTT class inserted")

if "self.twopttt = nn.ModuleDict()" not in s:
    s = s.replace("        self.ttt = nn.ModuleDict()",
                  "        self.ttt = nn.ModuleDict()\n        self.twopttt = nn.ModuleDict()", 1)
    print("twopttt ModuleDict added")

anchor2 = '''        if os.environ.get("LENA_TTT"):
            self.ttt[adapter_name] = TTTHead(self.in_features, self.out_features, r,
                                             lr=float(os.environ.get("LENA_TTT_LR", "1.0")))
            for p in self.ttt[adapter_name].parameters():
                p.requires_grad = True'''
add2 = '''
        if os.environ.get("LENA_2PTTT"):
            self.twopttt[adapter_name] = TwoPathTTT(self.in_features, self.out_features, r,
                                                    lr=float(os.environ.get("LENA_2PTTT_LR", "1.0")))
            for p in self.twopttt[adapter_name].parameters():
                p.requires_grad = True'''
if "self.twopttt[adapter_name] = TwoPathTTT" not in s:
    assert anchor2 in s, "ttt creation anchor missing"
    s = s.replace(anchor2, anchor2 + add2, 1)
    print("twopttt creation wired")

# forward injection: after the ttt injection block (uses layer input x)
anchor3 = '''        if name in self.ttt:
            tt = self.ttt[name]
            if tt.A.device != x.device:
                tt.to(x.device)
            dz = dz + tt(x).to(dz.dtype)'''
add3 = '''
        if name in self.twopttt:
            tp = self.twopttt[name]
            if tp.Wq.device != x.device:
                tp.to(x.device)
            dz = dz + tp(x).to(dz.dtype)'''
if "if name in self.twopttt:" not in s:
    assert anchor3 in s, "ttt forward anchor missing"
    s = s.replace(anchor3, anchor3 + add3, 1)
    print("twopttt forward injected")

open(F, "w").write(s)
ast.parse(open(F).read())
print("syntax OK")
