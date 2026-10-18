import ast, re
F = "peft/tuners/lena/layer.py"
s = open(F).read()

# Replace the whole span [class TTTHead ... before class SteerHead] with simplified v2 classes.
new_block = '''class TTTHead(nn.Module):
    """Simplified TTT: cross-token similarity mixing of the low-rank code (one gram matrix).
    k = A x (d->r);  c_i = (1/N) sum_j <k_j,k_i> k_j = (1/N) k (K^T K)_i ;  delta = gamma * B_t c.
    This is the closed form of ONE test-time step of a linear inner model, stripped of all
    machinery (no SwiGLU/hand-grad/Q,K,V) -- token i's code is a similarity-weighted average of
    the whole sequence's codes. Non-causal (full-sequence gram): judge by LIKELIHOOD eval, where
    it is fully active (no autoregressive-decode mismatch). gamma init 0 => exact base/LoRA start.
    Cheap: gram is r x r. Output in col(B_t)."""
    def __init__(self, d_in, d_out, r, lr=1.0):
        super().__init__()
        self.A = nn.Parameter(torch.randn(int(r), int(d_in)) * (float(d_in) ** -0.5))
        self.Bt = nn.Parameter(torch.randn(int(d_out), int(r)) * (float(r) ** -0.5))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        sq = False
        if x.dim() == 2:
            x = x.unsqueeze(0); sq = True
        xf = x.to(self.A.dtype)
        k = torch.matmul(xf, self.A.t())                 # [B,N,r]
        N = k.shape[-2]
        gram = torch.matmul(k.transpose(-2, -1), k) / float(N)   # [B,r,r]
        c = torch.matmul(k, gram)                        # [B,N,r]  cross-token mixed code
        delta = torch.matmul(c, self.Bt.t())             # [B,N,d_out]
        out = self.gamma.to(delta.dtype) * delta
        if sq:
            out = out.squeeze(0)
        return out.to(x.dtype)


class OutGate(nn.Module):
    """Input-dependent Householder REFLECTION of the LoRA write (norm-preserving, stable).
    delta = dz - 2 v_hat (v_hat . dz),  v = Wv z (per-input hyperplane normal), v_hat = v/||v||.
    Orthogonal transform: re-points the write direction per input WITHOUT changing its norm --
    escapes col(B) (reflected vectors vary with x) but cannot blow up (unlike additive steer) and
    is not a mere rescale (unlike the failed multiplicative gate). Wv init 0 => v=0 => no
    reflection => exact LoRA start."""
    def __init__(self, d_out, r):
        super().__init__()
        self.Wv = nn.Parameter(torch.zeros(int(d_out), int(r)))

    def forward(self, z, dz):
        v = torch.matmul(z.to(self.Wv.dtype), self.Wv.transpose(0, 1)).to(dz.dtype)  # [...,d_out]
        vhat = v / (v.norm(dim=-1, keepdim=True) + 1e-6)
        proj = (vhat * dz).sum(dim=-1, keepdim=True)
        return dz - 2.0 * vhat * proj


'''

start = s.index("class TTTHead(nn.Module):")
end = s.index("class SteerHead(nn.Module):")
assert start < end, "anchors out of order"
s = s[:start] + new_block + s[end:]

open(F, "w").write(s)
ast.parse(open(F).read())
print("v2 classes replaced; syntax OK")
