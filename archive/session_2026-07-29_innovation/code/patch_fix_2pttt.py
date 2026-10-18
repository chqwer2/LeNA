import ast
F = "peft/tuners/lena/layer.py"
s = open(F).read()

old = (
"        e = -v * self.scale\n"
"        ea = e * a\n"
"        eb = e * z1 * (sig * (1.0 + z2 * (1.0 - sig)))\n"
"        # causal cumulative inner-gradient states (position i sees only j<=i):\n"
"        S1 = torch.cumsum(torch.einsum('bnr,bns->bnrs', k, ea), dim=1)   # [B,N,r,r]\n"
"        S2 = torch.cumsum(torch.einsum('bnr,bns->bnrs', k, eb), dim=1)\n"
"        qw1 = torch.matmul(q, self.w1) - self.lr * torch.einsum('bnr,bnrs->bns', q, S1)\n"
"        qw2 = torch.matmul(q, self.w2) - self.lr * torch.einsum('bnr,bnrs->bns', q, S2)"
)
new = (
"        e = -v * self.scale\n"
"        ea = e * a\n"
"        eb = e * z1 * (sig * (1.0 + z2 * (1.0 - sig)))\n"
"        # causal cumulative inner-gradient states (position i sees only j<=i):\n"
"        Nn = k.shape[-2]\n"
"        cnt = torch.arange(1, Nn + 1, device=k.device, dtype=k.dtype).view(1, Nn, 1, 1)\n"
"        # ViT3 fix 1: prefix-MEAN gradient (ViT3 divides by N; causal => divide by i) -- keeps the\n"
"        # inner update from exploding as the prefix grows.\n"
"        S1 = torch.cumsum(torch.einsum('bnr,bns->bnrs', k, ea), dim=1) / cnt\n"
"        S2 = torch.cumsum(torch.einsum('bnr,bns->bnrs', k, eb), dim=1) / cnt\n"
"        # ViT3 fix 2: gradient CLIPPING g/(||g||+1) 'for stability' (per position, over dim -2).\n"
"        S1 = S1 / (S1.norm(dim=-2, keepdim=True) + 1.0)\n"
"        S2 = S2 / (S2.norm(dim=-2, keepdim=True) + 1.0)\n"
"        qw1 = torch.matmul(q, self.w1) - self.lr * torch.einsum('bnr,bnrs->bns', q, S1)\n"
"        qw2 = torch.matmul(q, self.w2) - self.lr * torch.einsum('bnr,bnrs->bns', q, S2)"
)
assert old in s, "causal_swiglu body anchor not found"
assert "ViT3 fix 1" not in s, "already fixed"
s = s.replace(old, new, 1)
open(F, "w").write(s)
ast.parse(open(F).read())
print("2pttt causal SwiGLU FIXED: added 1/i prefix normalization + gradient clipping (ViT3 stability)")
