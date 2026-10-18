import ast
F = "peft/tuners/lena/layer.py"
s = open(F).read()

# 1) TwoPathTTT: add a shared-code path. When forward gets a precomputed r-dim code z (= LoRA's A x),
#    reuse it as q=k=v (skip Wq/Wk/Wv) -> the TTT refines LoRA's OWN code (param-efficient, grounded).
old_fwd = '''    def forward(self, x):
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
        return delta.to(x.dtype)'''
new_fwd = '''    def forward(self, x, z=None):
        sq = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            if z is not None:
                z = z.unsqueeze(0)
            sq = True
        if z is not None:
            # SHARED-CODE mode: reuse LoRA's own r-dim code z as q=k=v (no Wq/Wk/Wv) -> TTT refines
            # LoRA's learned code. Param-efficient + semantically grounded.
            zc = z.to(self.w1.dtype)
            q = k = v = zc
            o1 = self._causal_swiglu(q, k, v)
            o2 = self._causal_conv(zc)
        else:
            xf = x.to(self.Wq.dtype)
            q = torch.matmul(xf, self.Wq.t()); k = torch.matmul(xf, self.Wk.t()); v = torch.matmul(xf, self.Wv.t())
            o1 = self._causal_swiglu(q, k, v)
            o2 = self._causal_conv(torch.matmul(xf, self.Wq2.t()))
        o = torch.cat([o1, o2], dim=-1)                           # [B,N,2r]
        delta = torch.matmul(o, self.Bt.t())
        if sq:
            delta = delta.squeeze(0)
        return delta.to(x.dtype)'''
assert old_fwd in s, "TwoPathTTT.forward anchor not found"
s = s.replace(old_fwd, new_fwd, 1)
print("TwoPathTTT shared-code path added")

# 2) forward injection: pass z when LENA_2PTTT_SHAREZ is set
old_inj = '''        if name in self.twopttt:
            tp = self.twopttt[name]
            if tp.Wq.device != x.device:
                tp.to(x.device)
            dz = dz + tp(x).to(dz.dtype)'''
new_inj = '''        if name in self.twopttt:
            tp = self.twopttt[name]
            if tp.Wq.device != x.device:
                tp.to(x.device)
            if os.environ.get("LENA_2PTTT_SHAREZ"):
                dz = dz + tp(x, z=z).to(dz.dtype)   # reuse LoRA code z as q=k=v
            else:
                dz = dz + tp(x).to(dz.dtype)'''
assert old_inj in s, "twopttt forward injection anchor not found"
s = s.replace(old_inj, new_inj, 1)
print("shared-code injection wired")

open(F, "w").write(s)
ast.parse(open(F).read())
print("syntax OK")
