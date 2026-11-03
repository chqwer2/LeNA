import json, random
d = json.load(open("/mnt/scratch2/users/hchen/data/lena_v7/pissa-dataset/metamath/test.json"))
A = "/mnt/scratch2/users/hchen/FLoRA/Experiments/lena_v7/audit/"
for t in ("gsm8k", "math"):
    L = [x for x in d if x["type"] == t]
    pick = [L[i] for i in sorted(random.Random(0).sample(range(len(L)), min(500, len(L))))]   # = eval_math --limit 500 --sample_seed 0
    json.dump(pick, open(A + f"{t}500_s0.json", "w")); print(t, len(pick))
