"""Paired bootstrap CIs (LeNA-LoRA, AuroRA-LoRA) for LLaMA3-8B math and commonsense, from per-item outputs."""
import json, os, random, sys
B = 2000
root = "/mnt/scratch2/users/hchen/FLoRA/Experiments/lena_v7"
def boot(a, b):
    n = len(a); rng = random.Random(0); d = []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        d.append(sum(a[i] - b[i] for i in idx) / n)
    d.sort(); return d[int(.025 * B)], d[int(.975 * B)]
rows = []
def emit(task, items):  # items: method -> {key: correct}
    keys = sorted(set.intersection(*[set(v) for v in items.values()]))
    acc = {m: sum(v[k] for k in keys) / len(keys) for m, v in items.items()}
    for m in ("lena", "aurora"):
        if m not in items: continue
        a = [items[m][k] for k in keys]; b = [items["lora"][k] for k in keys]
        lo, hi = boot(a, b)
        rows.append(dict(task=task, comparison=f"{m}-lora", n=len(keys), acc_method=round(100*acc[m], 2),
                         acc_lora=round(100*acc["lora"], 2), delta=round(100*(acc[m]-acc["lora"]), 2),
                         ci_lo=round(100*lo, 2), ci_hi=round(100*hi, 2)))
# math
mr = f"{root}/math/runs"
for tag, task in [("responses_s0.json", None), ("responses_ood_svamp_s0.json", "SVAMP"),
                  ("responses_ood_gsmhard_s0.json", "GSM-Hard"), ("responses_ood_asdiv_s0.json", "ASDiv")]:
    per = {}
    for m in ("lora", "lena", "aurora"):
        r = json.load(open(f"{mr}/math_{m}/{tag}"))
        per[m] = r
    if task is None:
        for ty, name in [("gsm8k", "GSM8K"), ("math", "MATH")]:
            emit(name, {m: {(x["instruction"]): int(bool(x["correct"])) for x in r if x.get("type", "gsm8k") == ty} for m, r in per.items()})
    else:
        emit(task, {m: {x["instruction"]: int(bool(x["correct"])) for x in r} for m, r in per.items()})
# commonsense, 7 tasks pooled (HellaSwag excluded: LoRA outputs 74/500 junk strings there)
cr = f"{root}/commonsense/runs"
tasks = ["boolq", "piqa", "social_i_qa", "winogrande", "ARC-Challenge", "ARC-Easy", "openbookqa"]
pooled = {}
for m in ("lora", "lena", "aurora"):
    d = {}; summ = json.load(open(f"{cr}/llama3_{m}/eval/summary.json"))
    for t in tasks:
        r = json.load(open(f"{cr}/llama3_{m}/eval/{t}.json"))
        c = [int(x["pred"] == x["answer"]) for x in r]
        s = summ[t]["acc"] if isinstance(summ[t], dict) else summ[t]
        assert abs(sum(c) / len(c) - s) < 1e-9, (m, t, sum(c) / len(c), s)
        for i, x in enumerate(r): d[(t, x["instruction"])] = c[i]
    pooled[m] = d
emit("Commonsense (7 tasks)", pooled)
cols = list(rows[0])
print(",".join(cols))
for r in rows: print(",".join(str(r[c]) for c in cols))
