"""Collect runs/fd_*/result.json into one CSV and print adapter x target tables (selected checkpoint,
training-domain validation) for acc / qwk / f1 / auc, averaged over seeds.

  python collect_fundus.py [runs_glob] [out.csv]
"""
import glob
import json
import sys
from collections import defaultdict

pat = sys.argv[1] if len(sys.argv) > 1 else "runs/fd_*/result.json"
out = sys.argv[2] if len(sys.argv) > 2 else "fundus_results.csv"
rows = []
for f in sorted(glob.glob(pat)):
    r = json.load(open(f)); i = r["info"]
    row = dict(run=f.split("/")[-2], target=i["target"], adapter=i["adapter"], lr=i["lr"], seed=i["seed"],
               steps=i["steps"], limit=i["limit"], aug=i["aug"], n_train=i["n_train"], n_target=i["n_target"],
               selected_step=r["selected_step"])
    for pre, d in (("val", r["id_val"]), ("tgt", r["target"]), ("tgt_last", r["target_last"]),
                   ("tgt_zs", r["target_zeroshot"])):
        row.update({f"{pre}_{k}": round(v, 4) for k, v in d.items()})
    for c, v in r.get("overrides", {}).items():
        row.update({f"tgt_{c}_{k}": round(x, 4) for k, x in v["target"].items()})
    row.update(s_per_step=round(r["s_per_step"], 3), wall_h=round(r["wall_s"] / 3600, 2))
    rows.append(row)
cols = []
for r in rows:
    cols += [k for k in r if k not in cols]
with open(out, "w") as fh:
    fh.write(",".join(cols) + "\n")
    for r in rows:
        fh.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
print(f"{len(rows)} runs -> {out}")
full = [r for r in rows if not r["limit"]]
tg = sorted({r["target"] for r in full}); ad = sorted({r["adapter"] for r in full})
for m in ("acc", "qwk", "f1", "auc"):
    acc = defaultdict(list)
    for r in full:
        acc[(r["adapter"], r["target"])].append(r[f"tgt_{m}"])
    print(f"\ntarget {m} (training-domain validation), mean over seeds [n]")
    print("adapter".ljust(8), *[t[:10].rjust(11) for t in tg], "avg".rjust(7))
    for a in ad:
        v = [sum(acc[(a, t)]) / len(acc[(a, t)]) if acc.get((a, t)) else None for t in tg]
        cells = [(f"{100 * x:.1f}[{len(acc[(a, t)])}]" if x is not None else "-").rjust(11) for x, t in zip(v, tg)]
        avg = f"{100 * sum(v) / len(v):.1f}" if v and all(x is not None for x in v) else "-"
        print(a.ljust(8), *cells, avg.rjust(7))
