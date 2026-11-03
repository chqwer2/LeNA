"""Register the AuroRA and LeNA-on-AuroRA adapters in the language training scripts."""
p = "lena_v7.py"; s = open(p).read()
old = 'ADAPTERS = {"lena": LeNALinear, "lora": LoRALinear, "aurora": AuroRALinear}'
assert old in s
s = s.replace(old, old + '\nADAPTERS["lena_aurora"] = lambda base, **kw: LeNALinear(base, base_nl="aurora", **kw)   # LeNA on AuroRA')
open(p, "w").write(s)
p = "commonsense/finetune_lena.py"; s = open(p).read()
s = s.replace('p.add_argument("--adapter", choices=["lena", "lora"], default="lena")', 'p.add_argument("--adapter", choices=["lena", "lora", "aurora", "lena_aurora"], default="lena")')
old = '''    if args.adapter == "lora":
        return dict(r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout, dora=args.dora)
'''
assert old in s
s = s.replace(old, old + '''    if args.adapter == "aurora":
        return dict(r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
''')
open(p, "w").write(s)
p = "math/finetune_math.py"; s = open(p).read()
s = s.replace('p.add_argument("--adapter", choices=["lena", "lora"], default="lena")', 'p.add_argument("--adapter", choices=["lena", "lora", "aurora", "lena_aurora"], default="lena")')
s = s.replace('''    if a.adapter == "lena":
        kw.update(K=a.knots''', '''    if a.adapter in ("lena", "lena_aurora"):
        kw.update(K=a.knots''')
open(p, "w").write(s)
print("ok")
