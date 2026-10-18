import ast
F = "peft/tuners/lena/layer.py"
s = open(F).read()

# 1) module-level causal running mean (leak-free "global context so far")
if "def _causal_mean(" not in s:
    fn = '''def _causal_mean(x):
    """Causal running mean over the sequence dim (-2): xbar_i = mean(x_1..x_i). Leak-free
    'global context so far'. Decode note: with KV-cache a single-token forward sees N=1 so
    xbar=x (global pathway sees the token, local sees ~0); judge two-pathway by LIKELIHOOD eval
    (full sequence) where it is fully active."""
    if x.dim() < 2:
        return x
    cs = x.cumsum(dim=-2)
    n = x.shape[-2]
    cnt = torch.arange(1, n + 1, device=x.device, dtype=cs.dtype).view(-1, 1)
    return cs / cnt


'''
    anchor = "class LeNALinear(nn.Module):"
    assert anchor in s
    s = s.replace(anchor, fn + anchor, 1)
    print("_causal_mean added")

# 2) ModuleDicts for the global pathway
if "self.lora_Ag = nn.ModuleDict()" not in s:
    s = s.replace("        self.ttt = nn.ModuleDict()",
                  "        self.ttt = nn.ModuleDict()\n        self.lora_Ag = nn.ModuleDict()\n        self.lora_Bg = nn.ModuleDict()", 1)
    print("lora_Ag/Bg ModuleDicts added")

# 3) env-gated creation of the global pathway (after local B requires_grad block)
anchor2 = '''        for p in self.lora_B[adapter_name].parameters():
            p.requires_grad = True'''
add2 = '''
        if os.environ.get("LENA_TWOPATH"):
            Ag = nn.Linear(self.in_features, r, bias=False)
            Bg = nn.Linear(r, self.out_features, bias=False)
            nn.init.xavier_uniform_(Ag.weight); nn.init.zeros_(Bg.weight)
            self.lora_Ag[adapter_name] = Ag
            self.lora_Bg[adapter_name] = Bg
            for p in Ag.parameters(): p.requires_grad = True
            for p in Bg.parameters(): p.requires_grad = True'''
if "if os.environ.get(\"LENA_TWOPATH\"):" not in s:
    assert anchor2 in s, "local-B requires_grad anchor missing"
    s = s.replace(anchor2, anchor2 + add2, 1)
    print("global pathway creation wired")

# 4) _phi helper method (extracted nonlinear core) inserted before forward
if "def _phi(self, name, z, act, gate):" not in s:
    phi = '''    def _phi(self, name, z, act, gate):
        """z -> h : the AuroRA-style nonlinear-code step (norm -> act -> gated linear/nonlinear
        interpolation), shared by both pathways so each is 'like AuroRA'."""
        if getattr(act, "kind", None) == "identity":
            return z
        if self.use_norm_before_act:
            norm = self.norm_before_act[name]
            nw = getattr(norm, "weight", None)
            zc = (norm(z.to(nw.dtype)).to(z.dtype) if nw is not None else norm(z))
        else:
            zc = z
        z_hwc, _, orig_ndim = _to_hwc(zc)
        phi = _from_hwc(act(z_hwc), orig_ndim).to(z.dtype)
        phi = phi.clamp(-50.0, 50.0)
        g = self._gate_value(gate, z)
        if g is not None:
            g = g.to(z.dtype)
        return phi if g is None else z + g * (phi - z)

    def forward(self, x: torch.Tensor, adapter_name: Optional[str] = None) -> torch.Tensor:'''
    old_fwd = "    def forward(self, x: torch.Tensor, adapter_name: Optional[str] = None) -> torch.Tensor:"
    assert s.count(old_fwd) == 1, "forward signature not unique"
    s = s.replace(old_fwd, phi, 1)
    print("_phi helper added")

# 5) replace the inline z->h->dz core with two-pathway-aware version
start = s.index("        z = A(drop(x).to(A.weight.dtype))")
end = s.index("        dz = B(h)") + len("        dz = B(h)")
new_core = '''        # Two-pathway (LENA_TWOPATH): global = phi on the causal-mean context (captures the
        # shared/task-level structure), local = phi on the per-token deviation from that mean
        # (captures the individual token). Both AuroRA-style (shared _phi). Bg init 0 => two-path
        # starts at exact base.
        if name in self.lora_Ag:
            xbar = _causal_mean(x)
            z = A(drop(x - xbar).to(A.weight.dtype))
            dz = B(self._phi(name, z, act, gate))
            Ag = self.lora_Ag[name]; Bg = self.lora_Bg[name]
            zg = Ag(drop(xbar).to(Ag.weight.dtype))
            dz = dz + Bg(self._phi(name, zg, act, gate))
        else:
            z = A(drop(x).to(A.weight.dtype))
            dz = B(self._phi(name, z, act, gate))'''
s = s[:start] + new_core + s[end:]
print("two-pathway forward core installed")

open(F, "w").write(s)
ast.parse(open(F).read())
print("syntax OK")
