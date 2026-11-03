"""Collect runs/db_*/result.json into one CSV (one row per run) and print per-dataset target-accuracy tables
(adapter x target domain, plus average), averaged over seeds.

  python collect_domainbed.py [runs_glob] [out.csv]
"""
import glob
import json
import sys
from collections import defaultdict

pat = sys.argv[1] if len(sys.argv) > 1 else "runs/db_*/result.json"
out = sys.argv[2] if len(sys.argv) > 2 else "domainbed_results.csv"
rows = []
for f in sorted(glob.glob(pat)):
    r = json.load(open(f)); i = r["info"]
    row = dict(run=f.split("/")[-2], dataset=i["dataset"], target=i["target"], adapter=i["adapter"], lr=i["lr"],
               seed=i["seed"], steps=i["steps"], limit=i["limit"], selected_step=r["selected_step"], id_val=r["id_val"],
               target_acc=r["target"], target_last=r["target_last"], target_zeroshot=r["target_zeroshot"],
               s_per_step=round(r["s_per_step"], 3), wall_h=round(r["wall_s"] / 3600, 2))
    for c, v in r.get("overrides", {}).items():
        row[f"target_{c}"] = v["target"]
    rows.append(row)
cols = []
for r in rows:
    cols += [k for k in r if k not in cols]
with open(out, "w") as fh:
    fh.write(",".join(cols) + "\n")
    for r in rows:
        fh.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
print(f"{len(rows)} runs -> {out}")
for ds in sorted({r["dataset"] for r in rows if not r["limit"]}):
    acc = defaultdict(list)
    for r in rows:
        if r["dataset"] == ds and not r["limit"]:
            acc[(r["adapter"], r["target"])].append(r["target_acc"])
    tg = sorted({t for _, t in acc}); ad = sorted({a for a, _ in acc})
    print(f"\n{ds} target acc (training-domain validation), mean over seeds [n]")
    print("adapter".ljust(8), *[t[:10].rjust(11) for t in tg], "avg".rjust(7))
    for a in ad:
        m = [sum(acc[(a, t)]) / len(acc[(a, t)]) if acc.get((a, t)) else None for t in tg]
        cells = [(f"{100 * v:.1f}[{len(acc[(a, t)])}]" if v is not None else "-").rjust(11) for v, t in zip(m, tg)]
        avg = f"{100 * sum(m) / len(m):.1f}" if all(v is not None for v in m) else "-"
        print(a.ljust(8), *cells, avg.rjust(7))
