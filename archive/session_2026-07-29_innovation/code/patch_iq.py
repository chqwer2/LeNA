import ast
F = "peft/tuners/lena/layer.py"
s = open(F).read()

# 1) ModuleDicts for the 2nd down-proj + lambda gate on the quadratic term
if "self.lora_A2 = nn.ModuleDict()" not in s:
    s = s.replace("        self.lora_Ag = nn.ModuleDict()\n        self.lora_Bg = nn.ModuleDict()",
                  "        self.lora_Ag = nn.ModuleDict()\n        self.lora_Bg = nn.ModuleDict()\n"
                  "        self.lora_A2 = nn.ModuleDict()\n        self.iq_lam = nn.ParameterDict()", 1)
    print("IQ ModuleDicts added")

# 2) env-gated creation (after local-B requires_grad)
anchor = '''        for p in self.lora_B[adapter_name].parameters():
            p.requires_grad = True'''
add = '''
        if os.environ.get("LENA_IQ"):
            A2 = nn.Linear(self.in_features, r, bias=False)
            nn.init.xavier_uniform_(A2.weight)
            self.lora_A2[adapter_name] = A2
            self.iq_lam[adapter_name] = nn.Parameter(torch.zeros(1))  # lambda init 0 => exact LoRA start
            for p in A2.parameters(): p.requires_grad = True'''
if "self.lora_A2[adapter_name] = A2" not in s:
    assert anchor in s, "local-B anchor missing"
    s = s.replace(anchor, anchor + add, 1)
    print("IQ creation wired")

# 3) forward: in the plain (non-two-path) branch, replace the linear code z with the quadratic code
#    z_iq = z + lam * (z ⊙ (A2 x))  BEFORE the nonlinearity/B. Input-dependent QUADRATIC (not a gate).
old_plain = '''        else:
            z = A(drop(x).to(A.weight.dtype))
            dz = B(self._phi(name, z, act, gate))'''
new_plain = '''        else:
            z = A(drop(x).to(A.weight.dtype))
            if name in self.lora_A2:
                A2 = self.lora_A2[name]
                if A2.weight.device != x.device:
                    A2.to(x.device); self.iq_lam[name].data = self.iq_lam[name].data.to(x.device)
                z2 = A2(drop(x).to(A2.weight.dtype))
                z = z + self.iq_lam[name].to(z.dtype) * (z * z2)   # input-dependent quadratic code
            dz = B(self._phi(name, z, act, gate))'''
if "if name in self.lora_A2:" not in s:
    assert old_plain in s, "plain forward branch anchor missing"
    s = s.replace(old_plain, new_plain, 1)
    print("IQ forward wired")

open(F, "w").write(s)
ast.parse(open(F).read())
print("syntax OK")
