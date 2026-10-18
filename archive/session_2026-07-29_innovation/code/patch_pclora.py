import ast
F = "peft/tuners/lena/layer.py"
s = open(F).read()

# ParamDict for pclora lambda
if "self.pclora_lam = nn.ParameterDict()" not in s:
    s = s.replace(
        "        self.lora_A2 = nn.ModuleDict()\n        self.iq_lam = nn.ParameterDict()",
        "        self.lora_A2 = nn.ModuleDict()\n        self.iq_lam = nn.ParameterDict()\n        self.pclora_lam = nn.ParameterDict()", 1)
    print("pclora ParamDict added")

# env-gated creation (before the IQ creation block)
anc = '        if os.environ.get("LENA_IQ"):'
add = ('        if os.environ.get("LENA_PCLORA"):\n'
       '            self.pclora_lam[adapter_name] = nn.Parameter(torch.zeros(1))  # lam init 0 => exact LoRA start\n'
       '        if os.environ.get("LENA_IQ"):')
if "self.pclora_lam[adapter_name]" not in s:
    assert anc in s, "IQ creation anchor missing"
    s = s.replace(anc, add, 1)
    print("pclora creation wired")

# forward: modulate the code z in the plain branch (after the IQ block, before dz=B(...))
anc2 = ('                z = z + self.iq_lam[name].to(z.dtype) * (z * z2)   # input-dependent quadratic code\n'
        '            dz = B(self._phi(name, z, act, gate))')
add2 = ('                z = z + self.iq_lam[name].to(z.dtype) * (z * z2)   # input-dependent quadratic code\n'
        '            if name in self.pclora_lam:\n'
        '                # Prompt-conditioned (TTT idea, decode-safe): causal-prefix associative-memory\n'
        '                # fit MODULATING the code. corr_i = (1/i) sum_{j<=i} <z_j,z_i> z_j =\n'
        '                # (1/i)(cumsum_j z_j z_j^T) z_i. Causal => decode-safe. lam init 0 => exact LoRA.\n'
        '                if z.dim() >= 3:\n'
        '                    zc = z.to(torch.float32)\n'
        "                    S = torch.cumsum(torch.einsum('bnr,bns->bnrs', zc, zc), dim=1)\n"
        '                    n = zc.shape[-2]\n'
        '                    cnt = torch.arange(1, n + 1, device=zc.device, dtype=torch.float32).view(-1, 1)\n'
        "                    corr = torch.einsum('bnr,bnrs->bns', zc, S) / cnt\n"
        '                    z = z + self.pclora_lam[name].to(z.dtype) * corr.to(z.dtype)\n'
        '            dz = B(self._phi(name, z, act, gate))')
if "if name in self.pclora_lam:" not in s:
    assert anc2 in s, "plain branch anchor missing"
    s = s.replace(anc2, add2, 1)
    print("pclora forward wired")

open(F, "w").write(s)
ast.parse(open(F).read())
print("syntax OK")
