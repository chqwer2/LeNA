"""R3 '--qt' quantile-transport LeNA layer (opt-in; lena_v7.py untouched; registered as adapter "lena_qt").

Per module, input and rank channel:
  Q_x(tau_k), tau_k = (k - 0.5)/Kq, Kq = 16: quantiles of u = A1 x over the input's tokens (weighted by w);
      causal: quantiles of the exclusive prefix (tokens < t) read from a soft histogram on the fixed knot grid.
  Q_ref: EMA (beta 0.99) of the batch-mean Q_x over training inputs (buffer, frozen in eval).
  T_x: piecewise-linear, T_x(Q_x(tau_k)) = Q_ref(tau_k), clamped outside [Q_x(tau_1), Q_x(tau_Kq)].
  u_hat = u + s (T_x(u) - u), s = n / (n + kappa), n = token count (causal: t), kappa = softplus(param) init 16.
  h = W x + (alpha / r) B rho(u_hat), rho the learned shared spline. A2 / v and the Stein fit are unused.
Overrides: lena_override(mode="fixed") -> s = 0 (plain fixed-curve adapter rho(u)); "plain" -> s = 1.
log_stats: s, F := mean_k |Q_x - Q_ref| / sd_k(Q_ref) (shift evidence), theta = rho (for the curve logs).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import lena_v7 as LV

KQ = 16


def taus(n, device, dtype):
    return (torch.arange(1, n + 1, device=device, dtype=dtype) - 0.5) / n


def interp_clamped(x, xp, fp):
    """Piecewise-linear interpolation, constant (clamped) outside [xp_0, xp_-1]. x (..., N); xp, fp (..., Q)."""
    Q = xp.shape[-1]
    idx = torch.searchsorted(xp.contiguous(), x.contiguous()).clamp(1, Q - 1)
    x0, x1 = xp.gather(-1, idx - 1), xp.gather(-1, idx)
    y0, y1 = fp.gather(-1, idx - 1), fp.gather(-1, idx)
    t = ((x - x0) / (x1 - x0).clamp_min(1e-6)).clamp(0, 1)
    return y0 + t * (y1 - y0)


def weighted_quantiles(u, w, nq=KQ):
    """u, w (B, N, r) -> (B, r, nq) quantiles over N (linear interpolation of the weighted empirical CDF)."""
    us, order = u.sort(1)
    ws = w.gather(1, order)
    cw = ws.cumsum(1)
    tot = cw[:, -1:].clamp_min(1e-12)
    mid = (cw - 0.5 * ws) / tot                                       # CDF at each sorted point (midpoint rule)
    t = taus(nq, u.device, u.dtype).view(1, nq, 1).expand(u.shape[0], nq, u.shape[2])
    q = interp_clamped(t.transpose(1, 2), mid.transpose(1, 2), us.transpose(1, 2))   # (B, r, nq)
    return q.cummax(-1).values


def prefix_quantiles(u, w, lo, hi, G, nq=KQ):
    """Causal: per position t the quantiles of tokens < t, from a soft (hat) histogram on G grid points in [lo, hi].
    Returns (B, N, r, nq) and the exclusive counts (B, N, r)."""
    j, lam = LV.hat_index(u.clamp(lo, hi), lo, hi, G)
    oh = F.one_hot(j, G).to(u.dtype) * ((1 - lam) * w).unsqueeze(-1) + F.one_hot(j + 1, G).to(u.dtype) * (lam * w).unsqueeze(-1)
    hist = F.pad(oh.cumsum(1), (0, 0, 0, 0, 1, 0))[:, :-1]            # exclusive prefix (B, N, r, G)
    cnt = hist.sum(-1)
    cdf = hist.cumsum(-1) / cnt.clamp_min(1e-12).unsqueeze(-1)       # CDF at the grid points
    grid = torch.linspace(lo, hi, G, device=u.device, dtype=u.dtype).expand_as(cdf)
    cdf = cdf + 1e-6 * torch.arange(G, device=u.device, dtype=u.dtype)          # strictly increasing for the inverse
    t = taus(nq, u.device, u.dtype).expand(*cdf.shape[:-1], nq)
    q = interp_clamped(t, cdf, grid)
    return q.cummax(-1).values, cnt


class LeNAQT(LV.LeNALinear):
    def __init__(self, base, kappa0=16.0, beta=0.99, **kw):
        super().__init__(base, **kw)
        r = self.A1.shape[0]
        self.kappa_raw = nn.Parameter(torch.tensor(math.log(math.expm1(kappa0)), device=self.A1.device))
        self.register_buffer("q_ref", torch.zeros(r, KQ, device=self.A1.device))
        self.register_buffer("q_ref_init", torch.zeros((), device=self.A1.device))
        self.beta = beta

    def _s_override(self, s):
        m = LV._CTX.force_mode
        return torch.zeros_like(s) if m == "fixed" else torch.ones_like(s) if m == "plain" else s

    def transport(self, u, w, causal):
        """Returns u_hat (B, N, r), s (B, r) mean, F (B, r)."""
        kap = F.softplus(self.kappa_raw)
        if not causal:
            qx = weighted_quantiles(u, w)                             # (B, r, Kq)
            if self.training:
                with torch.no_grad():
                    m = qx.detach().float().mean(0)
                    if self.q_ref_init.item() == 0:
                        self.q_ref.copy_(m); self.q_ref_init.fill_(1)
                    else:
                        self.q_ref.mul_(self.beta).add_((1 - self.beta) * m)
            ref = self.q_ref.to(u.dtype).expand_as(qx)
            T = interp_clamped(u.transpose(1, 2), qx, ref).transpose(1, 2)   # (B, N, r)
            n = w.sum(1)                                              # (B, r)
            s = self._s_override(n / (n + kap))
            uh = u + s.unsqueeze(1) * (T - u)
            Fq = (qx - ref).abs().mean(-1) / ref.std(-1).clamp_min(1e-6)
            return uh, s, Fq
        qx, cnt = prefix_quantiles(u, w, self.lo, self.hi, self.K)    # (B, N, r, Kq)
        if self.training:
            with torch.no_grad():
                m = qx[:, -1].detach().float().mean(0)               # the full-prefix quantiles of each input
                if self.q_ref_init.item() == 0:
                    self.q_ref.copy_(m); self.q_ref_init.fill_(1)
                else:
                    self.q_ref.mul_(self.beta).add_((1 - self.beta) * m)
        ref = self.q_ref.to(u.dtype).expand_as(qx)
        T = interp_clamped(u.unsqueeze(-1), qx, ref).squeeze(-1)      # (B, N, r)
        s = self._s_override(cnt / (cnt + kap))
        uh = u + s * (T - u)
        Fq = ((qx - ref).abs().mean(-1) / ref.std(-1).clamp_min(1e-6)).mean(1)
        return uh, s.mean(1), Fq

    def forward(self, x):
        squeeze = x.dim() == 2
        if squeeze:
            x = x.unsqueeze(0)
        xd = self.drop(x)
        sd = torch.promote_types(xd.dtype, torch.float32)
        u = F.linear(xd, self.A1.to(xd.dtype)).to(sd)
        mask = LV._CTX.mask
        Bsz, N, r = u.shape
        w = mask.to(sd).unsqueeze(-1).expand(Bsz, N, r) if mask is not None and tuple(mask.shape[:2]) == (Bsz, N) \
            else torch.ones_like(u)
        causal = self.causal if LV._CTX.causal is None else LV._CTX.causal
        uh, s, Fq = self.transport(u, w, causal)
        rho = self.rho.to(sd)
        j, lam = LV.hat_index(uh, self.lo, self.hi, self.K)
        c = LV.spline_eval(rho.view(1, 1, *rho.shape), j, lam)
        if self.log_stats:
            self.last = dict(s=s.detach(), F=Fq.detach(), p=torch.zeros_like(s), theta=rho.expand(Bsz, *rho.shape).detach(),
                             u=u.detach(), v=uh.detach())
        out = self.base(x) + self.scaling * F.linear(c.to(x.dtype), self.B.to(x.dtype))
        return out.squeeze(0) if squeeze else out


def register():
    LV.ADAPTERS["lena_qt"] = LeNAQT
