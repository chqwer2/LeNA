# TTT parallel-branch: design, variants, honest results, and the blocker

**Idea.** Instead of a static low-rank ΔW, attach a **parallel gated branch** that does test-time training (an inner-loop
weight update per input) on the residual stream, ViT³-grounded:
```
h_out = FrozenAttn(LN h) + γ · TwoPathTTT(LN h)
```
Only the branches train (base frozen). Trainable ≈ 50.6M at r=64 (0.745%). Inference needs `use_cache=False` (the TTT
needs the full prefix), so decoding is O(N²) — a known cost, fixable via a recurrent/stateful formulation (future work).

## Core module (TwoPathTTT): global causal SwiGLU inner-TTT + local causal conv
```python
class TwoPathTTT(nn.Module):
    def __init__(self, d_in, d_out, r, ksize=4):
        super().__init__()
        self.r=r; self.scale=r**-0.5; self.ksize=ksize
        self.Wq=nn.Parameter(torch.randn(r,d_in)*d_in**-0.5); self.Wk=nn.Parameter(torch.randn(r,d_in)*d_in**-0.5)
        self.Wv=nn.Parameter(torch.randn(r,d_in)*d_in**-0.5); self.Wq2=nn.Parameter(torch.randn(r,d_in)*d_in**-0.5)
        self.w1=nn.Parameter(trunc_normal(r,r,std=.02)); self.w2=nn.Parameter(trunc_normal(r,r,std=.02))
        self.conv=nn.Parameter(torch.zeros(r,1,ksize)); self.Bt=nn.Parameter(torch.zeros(d_out,2*r))  # Bt=0 => no-op init
    def _sg(self,q,k,v):                              # causal closed-form 1-step inner gradient, cumulative state
        z1=k@self.w1; z2=k@self.w2; sig=torch.sigmoid(z2); a=z2*sig
        e=-v*self.scale; ea=e*a; eb=e*z1*(sig*(1+z2*(1-sig)))
        N=k.shape[-2]; cnt=torch.arange(1,N+1).view(1,N,1,1)
        S1=torch.cumsum(einsum('bnr,bns->bnrs',k,ea),1)/cnt; S2=torch.cumsum(einsum('bnr,bns->bnrs',k,eb),1)/cnt
        S1=S1/(S1.norm(-2,keepdim=True)+1); S2=S2/(S2.norm(-2,keepdim=True)+1)      # prefix-mean + norm (heuristic)
        qw1=q@self.w1 - einsum('bnr,bnrs->bns',q,S1); qw2=q@self.w2 - einsum('bnr,bnrs->bns',q,S2)
        return qw1*silu(qw2)
    def _cv(self,q2):                                # local causal depthwise conv path
        xc=pad(q2.transpose(1,2),(self.ksize-1,0)); return conv1d(xc,self.conv,groups=self.r).transpose(1,2)
    def forward(self,x):
        q=x@self.Wq.t(); k=x@self.Wk.t(); v=x@self.Wv.t()
        return torch.cat([self._sg(q,k,v), self._cv(x@self.Wq2.t())],-1) @ self.Bt.t()
```

## Variants tried (each a file: ttt_branch*.py / ttt_c.py) and honest results
| variant | change | gsm8k | vs LoRA .33 |
|---|---|---|---|
| branch (attn only) | γ·TwoPathTTT added to attention out | 0.27 | worse |
| +MLP coverage | also wrap the MLP sublayer | ~0.27 (train-loss only −0.03) | worse |
| **v4 strengthened** | learnable per-layer `inner_lr` + per-channel gate + enriched core (r→2r→r, zero-init) | **0.30** | worse but best; gain is in *generation*, not fit |
| LoRA + branch (naive) | joint, γ scalar init 1 | 0.267 | branch drags LoRA down |
| **C: LoRA + gated branch** | per-channel gate + L1 "do-no-harm" | **0.24 (in) / 0.20 (OOD)** | worst — gate L1 did not stop the harm |

## The decisive diagnosis (why the whole line fails)
Train loss is good (C reaches 0.19 with LoRA) but eval is bad (0.24) → a **train/generation mismatch (exposure bias
amplified by test-time adaptation)**: training is teacher-forced on the *gold* answer, so the TTT state fits; at
autoregressive generation the TTT adapts to the model's own (possibly wrong) growing prefix and *amplifies* errors. This is
structural to TTT-at-inference, not a tuning issue. It explains why every branch config underperforms its training fit.

## Two known design weaknesses (leads for the fix)
1. **Prefix-mean `/cnt`** washes out the accumulating adaptation; a decaying/EMA state (Mamba/linear-attn style) keeps
   recent tokens meaningful and is train/inference consistent.
2. **`use_cache=False` O(N²)** — a recurrent stateful state removes this too. Both point to the same fix.

## Verdict
Not submittable as a positive result. The real path is a **recurrent/stateful TTT** consistent between train and
generation, then demonstrate an **OOD / test-time-adaptation** advantage LoRA structurally cannot have. Future work
(possible ICASSP once a positive result exists). Do not pile more tweaks on the teacher-forced version.
```
Note: also relevant — batched left-padded eval pollutes the cumulative causal state with pad tokens, so branch eval must be
bs=1 (unbatched). Static adapters (LoRA/LeNA/IQ) are unaffected.
```
