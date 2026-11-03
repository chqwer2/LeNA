import json
d = json.load(open("/mnt/scratch2/users/hchen/data/lena_v7/pissa-dataset/metamath/test.json"))
g = [x for x in d if x["type"] == "gsm8k"][:100]
json.dump(g, open("/mnt/scratch2/users/hchen/FLoRA/Experiments/lena_v7/audit/gsm100.json", "w"))
print(len(g))
