"""Per method/dataset/protocol accuracy + answerless count, and paired bootstrap 95% CI of LeNA-LoRA (10k resamples)."""
import json, os, sys
import numpy as np
root, tag = sys.argv[1], sys.argv[2]
rows = ["method,dataset,protocol,n,acc,answerless,delta_lena_minus_lora,ci95_lo,ci95_hi"]
for proto, sub in (("generation_beam4_mnt1", "langood_gen_"), ("loglik_scoring", "langood_score_")):
    for ds in ("commonsenseqa", "mmlu"):
        c = {}
        for m in ("lena", "lora"):
            r = json.load(open(os.path.join(root, f"runs/llama3_{m}", sub + tag, f"{ds}.json")))
            c[m] = np.array([x["pred"] == x["answer"] for x in r], float)
            nl = sum(x["pred"] == "" for x in r)
            rows.append(f"{m},{ds},{proto},{len(r)},{100 * c[m].mean():.1f},{nl},,,")
        d = c["lena"] - c["lora"]; rng = np.random.default_rng(0)
        bs = d[rng.integers(0, len(d), (10000, len(d)))].mean(1) * 100
        rows.append(f"lena-lora,{ds},{proto},{len(d)},,,{100 * d.mean():.1f},{np.percentile(bs, 2.5):.1f},{np.percentile(bs, 97.5):.1f}")
print("\n".join(rows))
