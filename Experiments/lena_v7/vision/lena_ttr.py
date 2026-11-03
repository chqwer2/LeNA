"""R12 'ttr' LeNA layer (opt-in; lena_v7.py untouched — registered as adapter "lena_ttr" in this process only).

Three changes to LeNALinear, everything else (A1 keys u, A2 values v, shared prior rho, B, scaling) as in lena_v7:
  1. cross-fit: non-causal inputs split their tokens into two halves (checkerboard on a square token grid, else
     alternating positions); statistics fitted on one half are applied to the other half, both ways. Causal inputs use
     the exclusive prefix (statistics of tokens < t).
  2. separate query: the curve is applied to q = A3 x (new rank-r projection, initialised like A1), not to u; u and v
     are used only for fitting, so the output cannot copy v.
  3. cross-validated shrinkage: theta = rho + s (theta_hat - rho), theta_hat the plain fit, s = clamp(R2_cv, 0, 1)
     detached, per input and channel, R2_cv = 1 - SSE(theta_hat) / SSE(rho) on the held-out half (causal: the
     prequential version over the prefix, each token predicted by the fit of the tokens before it).
     A fit that does not generalise gives s -> 0 and the shared curve rho.
Overrides: lena_override(mode="fixed") forces s = 0, mode="plain" forces s = 1.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import lena_v7 as LV


def halves(N, device):
    g = int(round(N ** 0.5))
    if g * g == N:
        i = torch.arange(g, device=device)
        return ((i[:, None] + i[None, :]) % 2 == 0).flatten()
    return torch.arange(N, device=device) % 2 == 0


def plain_fit(diag, off, m, rho, tau):
    rhs = m - LV.tri_mv(diag, off, rho)
    delta, _ = LV.retained_solve(diag, off, rhs, tau)
    return rho + delta


def curve(theta, z, lo, hi, K):
    """theta (B, r, K) per input or (B, N, r, K) per token; z (B, N, r)."""
    j, lam = LV.hat_index(z, lo, hi, K)
    if theta.dim() == 3:
        theta = theta.unsqueeze(1)
    return LV.spline_eval(theta, j, lam)


class LeNATTR(LV.LeNALinear):
    def __init__(self, base, **kw):
        super().__init__(base, **kw)
        assert self.magnitude is None and self.anl is None and self.prior_net is None, "ttr: plain LeNA only"
        self.A3 = nn.Parameter(torch.empty_like(self.A1))
        nn.init.xavier_uniform_(self.A3)

    def _s_override(self, s):
        m = LV._CTX.force_mode
        if m == "fixed":
            return torch.zeros_like(s)
        if m == "plain":
            return torch.ones_like(s)
        return s

    def codes_ttr(self, u, v, q, w, causal):
        K, lo, hi, tau = self.K, self.lo, self.hi, self.tau
        rho = self.rho.to(u.dtype)
        B, N, r = u.shape
        if not causal:
            a = halves(N, u.device).to(w.dtype)[None, :, None]
            thetas, ss = [], []
            for fit_w, test_w in ((w * a, w * (1 - a)), (w * (1 - a), w * a)):
                _, _, diag, off, m, e, nt = LV.pooled_stats(u, v, fit_w, lo, hi, K)
                th = plain_fit(diag, off, m, rho, tau)                         # (B, r, K)
                with torch.no_grad():
                    sse_f = ((v - curve(th, u, lo, hi, K)) ** 2 * test_w).sum(1)
                    sse_r = ((v - curve(rho.expand_as(th), u, lo, hi, K)) ** 2 * test_w).sum(1)
                    s = (1 - sse_f / sse_r.clamp_min(1e-12)).clamp(0, 1)
                s = self._s_override(s)
                thetas.append(rho + s.unsqueeze(-1) * (th - rho)); ss.append(s)
            c_from_a = curve(thetas[0], q, lo, hi, K)                          # fitted on A, applied to B tokens
            c_from_b = curve(thetas[1], q, lo, hi, K)
            c = torch.where(a.bool(), c_from_b, c_from_a)
            s_all = 0.5 * (ss[0] + ss[1])
            theta_mean = 0.5 * (thetas[0] + thetas[1])
        else:
            _, _, diag, off, m, e, nt = LV.prefix_stats(u, v, w, lo, hi, K)    # inclusive prefix (B, N, r, .)
            ex = lambda t: F.pad(t, (0,) * (2 * (t.dim() - 2)) + (1, 0))[:, :-1] if t.dim() > 2 else F.pad(t, (1, 0))[:, :-1]  # noqa: E731
            diag, off, m = ex(diag), ex(off), ex(m)                             # exclusive prefix: tokens < t
            th = plain_fit(diag, off, m, rho, tau)                             # (B, N, r, K)
            with torch.no_grad():
                err_f = (v - curve(th, u, lo, hi, K)) ** 2 * w                  # token t predicted from tokens < t
                err_r = (v - curve(rho.expand(B, r, K), u, lo, hi, K)) ** 2 * w
                cf = F.pad(err_f.cumsum(1), (0, 0, 1, 0))[:, :-1]               # prequential sums over tokens < t
                cr = F.pad(err_r.cumsum(1), (0, 0, 1, 0))[:, :-1]
                s = torch.where(cr > 0, (1 - cf / cr.clamp_min(1e-12)).clamp(0, 1), torch.zeros_like(cr))
            s = self._s_override(s)
            theta = rho + s.unsqueeze(-1) * (th - rho)
            c = curve(theta, q, lo, hi, K)
            s_all, theta_mean = s.mean(1), theta.mean(1)
        return c, dict(s=s_all, p=torch.zeros_like(s_all), F=torch.zeros_like(s_all), theta=theta_mean.detach(),
                       u=u.detach(), v=v.detach())

    def forward(self, x):
        squeeze = x.dim() == 2
        if squeeze:
            x = x.unsqueeze(0)
        xd = self.drop(x)
        sd = torch.promote_types(xd.dtype, torch.float32)
        u = F.linear(xd, self.A1.to(xd.dtype)).to(sd)
        v = F.linear(xd, self.A2.to(xd.dtype)).to(sd)
        q = F.linear(xd, self.A3.to(xd.dtype)).to(sd)
        mask = LV._CTX.mask
        Bsz, N, r = u.shape
        w = mask.to(sd).unsqueeze(-1).expand(Bsz, N, r) if mask is not None and tuple(mask.shape[:2]) == (Bsz, N) \
            else torch.ones_like(u)
        causal = self.causal if LV._CTX.causal is None else LV._CTX.causal
        c, d = self.codes_ttr(u, v, q, w, causal)
        if self.log_stats:
            self.last = {k: t.detach() for k, t in d.items()}
        out = self.base(x) + self.scaling * F.linear(c.to(x.dtype), self.B.to(x.dtype))
        return out.squeeze(0) if squeeze else out


def register():
    LV.ADAPTERS["lena_ttr"] = LeNATTR
