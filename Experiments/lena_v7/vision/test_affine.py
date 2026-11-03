import sys, torch; sys.path.insert(0, "."); sys.path.insert(0, "..")
import lena_v7 as LV, screen_affine as SA
torch.manual_seed(0); torch.set_default_dtype(torch.float64)
B, N, r, K = 3, 200, 2, 16
u = torch.randn(B, N, r) * 1.2; v = torch.randn(B, N, r) + 0.7 * torch.tanh(u)
w = torch.ones_like(u); rho = torch.randn(r, K).cumsum(-1) * 0.3
j, lam, diag, off, m, e, nt = LV.pooled_stats(u, v, w, -3.0, 3.0, K)
rb = rho.expand_as(m); ru = LV.spline_eval(rb.unsqueeze(1), j, lam)        # rho(u_i)
# affine LS on tokens vs closed form
X = torch.stack([ru, torch.ones_like(ru)], -1)                              # (B,N,r,2)
coef = torch.linalg.lstsq(X.permute(0, 2, 1, 3), v.permute(0, 2, 1).unsqueeze(-1)).solution.squeeze(-1)  # (B,r,2)
P = SA.proj_S(rb, torch.zeros_like(m), diag, off)
SA.G["variant"] = "affine"; SA.ORIG = lambda *a: (rb + 0.5, dict(s=torch.ones(B, r)))
th, _ = SA.fit_and_step(diag, off, m, e, nt, rho, torch.zeros(()), 1.0, "lena")
want = coef[..., :1] * rb + coef[..., 1:]
print("affine LS max err", (th - want).abs().max().item())
f = torch.randn(B, r, K); Pf = SA.proj_S(rb, f, diag, off)
print("idempotent err", (SA.proj_S(rb, Pf, diag, off) - Pf).abs().max().item())
print("orth err", (LV.tri_mv(diag, off, f - Pf) * rb).sum(-1).abs().max().item(), (LV.tri_mv(diag, off, f - Pf)).sum(-1).abs().max().item())
fu = LV.spline_eval(f.unsqueeze(1), j, lam); vt = SA.var_tok(f, diag, off, nt)
print("var_tok err", (vt - fu.var(1, unbiased=False)).abs().max().item())
