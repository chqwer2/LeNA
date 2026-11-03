"""Add LeNA-on-AuroRA: the shared curve is AuroRA's learned non-linearity g(u) (ANL), and LeNA fits a per-input
correction to the residual v - g(u) with a zero prior; code = g(u) + spline(s * Delta)(u). With s = 0 it is AuroRA."""
import re
p = "lena_v7.py"; s = open(p).read()
s = s.replace('''                 use_checkpoint=True, log_stats=False):
        super().__init__()
        assert mode in''', '''                 use_checkpoint=True, log_stats=False, base_nl=None):
        super().__init__()
        assert mode in''', 1)
old = '''        self.omega = nn.Parameter(torch.zeros((), device=dev), requires_grad=learn_omega)
'''
assert old in s
s = s.replace(old, old + '''        self.anl = None
        if base_nl == "aurora":                      # LeNA on AuroRA: learned shared non-linearity, per-input residual fit
            self.anl = ANL(r, r).to(dev)
            self.rho = nn.Parameter(torch.zeros(r, K, device=dev), requires_grad=False)
''', 1)
old2 = '''        if self.use_checkpoint and self.training and torch.is_grad_enabled():
            c, d = checkpoint(self._codes, u, v, w, use_reentrant=False)
        else:
            c, d = self._codes(u, v, w)
'''
assert old2 in s
s = s.replace(old2, '''        g = self.anl(u) if self.anl is not None else None
        vt = v - g if g is not None else v
        if self.use_checkpoint and self.training and torch.is_grad_enabled():
            c, d = checkpoint(self._codes, u, vt, w, use_reentrant=False)
        else:
            c, d = self._codes(u, vt, w)
        if g is not None:
            c = c + g
''', 1)
open(p, "w").write(s)
q = "vision/clip_fewshot.py"; t = open(q).read()
t = t.replace('choices=["lena", "lora", "dora", "aurora", "zeroshot"]', 'choices=["lena", "lena_aurora", "lora", "dora", "aurora", "zeroshot"]')
t = t.replace('''        if a.adapter == "lena":
            kw.update(K=16, causal=False, use_checkpoint=False, mode=a.mode)
''', '''        if a.adapter in ("lena", "lena_aurora"):
            kw.update(K=16, causal=False, use_checkpoint=False, mode=a.mode)
        if a.adapter == "lena_aurora":
            kw["base_nl"] = "aurora"
''')
t = t.replace('''adapter="lora" if a.adapter == "dora" else a.adapter, **kw)''', '''adapter={"dora": "lora", "lena_aurora": "lena"}.get(a.adapter, a.adapter), **kw)''')
t = t.replace('''    if a.adapter == "lena":
        conds += [''', '''    if a.adapter in ("lena", "lena_aurora"):
        conds += [''')
t = t.replace('''(c == "learned" and a.adapter == "lena")''', '''(c == "learned" and a.adapter in ("lena", "lena_aurora"))''')
open(q, "w").write(t)
print("patched", s.count("base_nl"), t.count("lena_aurora"))
