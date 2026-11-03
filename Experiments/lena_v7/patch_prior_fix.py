"""Option A for the shared prior (Kelvin copy): token thinning of the statistics during training (so the step s varies
and the prior receives gradient) and a separate learning rate (no weight decay) for rho and omega. Default off."""
import sys

root = sys.argv[1]


def patch(path, pairs):
    s = open(path).read()
    for a, b in pairs:
        assert s.count(a) == 1, (path, a[:60], s.count(a))
        s = s.replace(a, b, 1)
    open(path, "w").write(s)
    print("patched", path)


patch(f"{root}/lena_v7.py", [
    ('''        self.last = None
        self._state = None
''', '''        self.last = None
        self._state = None
        self.stat_drop = 0.0   # training only: each sample keeps a random fraction in [1 - stat_drop, 1] of its tokens
'''),
    ('''        rho, omega = self.rho.to(u.dtype), self.omega.to(u.dtype)
        if _CTX.force_prior == "identity":''', '''        rho, omega = self.rho.to(u.dtype), self.omega.to(u.dtype)
        if self.training and self.stat_drop > 0:       # thin the statistics so the step meets weak evidence too
            keep = 1 - self.stat_drop * torch.rand(w.shape[0], 1, 1, device=w.device, dtype=w.dtype)
            w = w * (torch.rand(w.shape[0], w.shape[1], 1, device=w.device, dtype=w.dtype) < keep).to(w.dtype)
        if _CTX.force_prior == "identity":'''),
])
patch(f"{root}/vision/clip_fewshot.py", [
    ('''    p.add_argument("--load", default="", help="adapter.pt of a finished run: skip training, evaluate only")''',
     '''    p.add_argument("--load", default="", help="adapter.pt of a finished run: skip training, evaluate only")
    p.add_argument("--stat_drop", type=float, default=0.0, help="LeNA: token thinning of the statistics in training")
    p.add_argument("--prior_lr_mult", type=float, default=1.0, help="LeNA: learning-rate factor for rho and omega (no decay)")'''),
    ('''        params = [q for q in model.parameters() if q.requires_grad]
        opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=1e-2, betas=(0.9, 0.999))''',
     '''        for L in lena_layers(model):
            L.stat_drop = a.stat_drop
        pri = [q for n, q in model.named_parameters() if q.requires_grad and (n.endswith(".rho") or n.endswith(".omega"))]
        rest = [q for n, q in model.named_parameters() if q.requires_grad and not (n.endswith(".rho") or n.endswith(".omega"))]
        groups = [dict(params=rest)] + ([dict(params=pri, lr=a.lr * a.prior_lr_mult, weight_decay=0.0)] if pri else [])
        opt = torch.optim.AdamW(groups, lr=a.lr, weight_decay=1e-2, betas=(0.9, 0.999))'''),
])
