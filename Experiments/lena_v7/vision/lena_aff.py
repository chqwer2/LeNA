"""H15 adapters (opt-in; lena_v7.py untouched). Registered at runtime: ADAPTERS["lena_aff"], ADAPTERS["lora_inorm"];
install("lena_aff") additionally maps adapter "lena" -> LeNAAff (and install("lora_inorm") maps "lora" ->
LoRAINorm) so the unmodified training scripts build them (see train_h15.py).

lena_aff   LeNA restricted to the affine family of the shared curve. Per input x (non-causal: its tokens; causal: the
           prefix <= t, as LeNA) and rank channel, with u = A1 x, v = A2 x and the same weighted spline statistics:
             (a_x, b_x) = least squares of v on [rho(u_i), 1]   (closed form from n, m: hat functions sum to one)
             s          = LeNA's Stein shrink of the same statistics (lena_v7.fit_and_step, mode / overrides as LeNA)
             c          = rho(u) + s (a_x rho(u) + b_x - rho(u)),   h = W x + (alpha / r) B c
           a_x is ridge-shrunk toward 1 by lam = 1e-4 mean(rho(u)^2) + 1e-6 (only matters when rho(u) is ~constant).
           Overrides: lena_override(mode="fixed") -> s = 0 -> c = rho(u) (fixed curve); "plain" -> s = 1.
lora_inorm LoRA with instance normalisation of the bottleneck: u = A1 x; z = (u - mean_tok(u)) / sqrt(var_tok(u) + eps)
           per input and channel (weighted by the padding mask; causal: inclusive prefix statistics);
           h = W x + (alpha / r) B (gamma * z + beta), gamma (init 1), beta (init 0) per channel.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import lena_v7 as LV


def _weights(u):
    mask = LV._CTX.mask
    B, N, r = u.shape
    if mask is not None and tuple(mask.shape[:2]) == (B, N):
        return mask.to(u.dtype).unsqueeze(-1).expand(B, N, r)
    return torch.ones_like(u)


class LeNAAff(LV.LeNALinear):
    def __init__(self, base, **kw):
        super().__init__(base, **kw)
        assert self.anl is None and self.prior_net is None and self.stat_net is None and self.magnitude is None, \
            "lena_aff supports the plain LeNA configuration only"

    def affine_theta(self, diag, off, m, ntok, rho):
        """Knot vector of the LS affine curve a rho + b (..., K); returns theta_aff, a, b."""
        rb = rho.expand_as(m)
        n = ntok.clamp_min(1)
        Nr = LV.tri_mv(diag, off, rb)
        mr = Nr.sum(-1) / n                                           # mean rho(u)    (1'N rho / n)
        mrr = (rb * Nr).sum(-1) / n                                   # mean rho(u)^2
        mv = m.sum(-1) / n                                            # mean v
        mrv = (rb * m).sum(-1) / n                                    # mean rho(u) v
        var, cov = (mrr - mr * mr).clamp_min(0), mrv - mr * mv
        lam = 1e-4 * mrr + 1e-6
        a = (cov + lam) / (var + lam)
        b = mv - a * mr
        return a.unsqueeze(-1) * rb + b.unsqueeze(-1), a, b

    def forward(self, x):
        squeeze = x.dim() == 2
        if squeeze:
            x = x.unsqueeze(0)
        xd = self.drop(x)
        sd = torch.promote_types(xd.dtype, torch.float32)
        u = F.linear(xd, self.A1.to(xd.dtype)).to(sd)
        v = F.linear(xd, self.A2.to(xd.dtype)).to(sd)
        w = _weights(u)
        causal = self.causal if LV._CTX.causal is None else LV._CTX.causal
        assert not (causal and LV._CTX.stream and not self.training), "lena_aff: KV-cache decoding not implemented"
        if self.training and self.stat_drop > 0:
            keep = 1 - self.stat_drop * torch.rand(w.shape[0], 1, 1, device=w.device, dtype=w.dtype)
            w = w * (torch.rand(w.shape[0], w.shape[1], 1, device=w.device, dtype=w.dtype) < keep).to(w.dtype)
        stats = LV.prefix_stats if causal else LV.pooled_stats
        j, lam, diag, off, m, e, ntok = stats(u, v, w, self.lo, self.hi, self.K)
        rho, omega = self.rho.to(sd), self.omega.to(sd)
        if LV._CTX.force_prior == "identity":
            rho = torch.linspace(self.lo, self.hi, self.K, device=u.device, dtype=sd).expand_as(rho)
        mode = LV._CTX.force_mode or self.mode
        _, d = LV.fit_and_step(diag, off, m, e, ntok, rho, omega, self.tau, mode)
        aff, a, b = self.affine_theta(diag, off, m, ntok, rho)
        rb = rho.expand_as(m)
        theta = rb + d["s"].unsqueeze(-1) * (aff - rb)
        c = LV.spline_eval(theta if causal else theta.unsqueeze(1), j, lam)
        if self.log_stats:
            self.last = {k: t.detach() for k, t in dict(d, theta=theta, u=u, v=v, a=a, b=b).items()}
        out = self.base(x) + self.scaling * F.linear(c.to(x.dtype), self.B.to(x.dtype))
        return out.squeeze(0) if squeeze else out


class LoRAINorm(LV.LoRALinear):
    def __init__(self, base, r=8, alpha=16, dropout=0.0, dora=False, eps=1e-5, **_):
        super().__init__(base, r=r, alpha=alpha, dropout=dropout, dora=False)
        assert not dora, "lora_inorm: no DoRA variant"
        dev = base.weight.device
        self.gamma = nn.Parameter(torch.ones(r, device=dev))
        self.beta = nn.Parameter(torch.zeros(r, device=dev))
        self.eps = eps

    def forward(self, x):
        squeeze = x.dim() == 2
        if squeeze:
            x = x.unsqueeze(0)
        xd = self.drop(x)
        sd = torch.promote_types(xd.dtype, torch.float32)
        u = F.linear(xd, self.A1.to(xd.dtype)).to(sd)
        w = _weights(u)
        causal = bool(LV._CTX.causal)
        if causal:
            n = w.cumsum(1).clamp_min(1e-6)
            mu = (w * u).cumsum(1) / n
            var = ((w * u * u).cumsum(1) / n - mu * mu).clamp_min(0)
        else:
            n = w.sum(1, keepdim=True).clamp_min(1e-6)
            mu = (w * u).sum(1, keepdim=True) / n
            var = (w * (u - mu) ** 2).sum(1, keepdim=True) / n
        z = (u - mu) / torch.sqrt(var + self.eps)
        c = self.gamma.to(sd) * z + self.beta.to(sd)
        out = self.base(x) + self.scaling * F.linear(c.to(x.dtype), self.B.to(x.dtype))
        return out.squeeze(0) if squeeze else out


def register():
    LV.ADAPTERS["lena_aff"] = LeNAAff
    LV.ADAPTERS["lora_inorm"] = LoRAINorm


def install(variant):
    """Make the unmodified training scripts build the H15 variant: lena_aff replaces "lena", lora_inorm "lora"."""
    register()
    if variant == "lena_aff":
        LV.ADAPTERS["lena"] = LeNAAff
    elif variant == "lora_inorm":
        LV.ADAPTERS["lora"] = LoRAINorm
    else:
        raise ValueError(variant)
