"""Camelyon17 summary: results table (model selection on ID validation per method), element checks of the selected LeNA
run (s = 0 / s = 1 / identity prior at evaluation), and the per-hospital fit statistics of the selected LeNA run.

  python plot_camelyon.py --runs <dir with cam_* run folders> --out <dir> --tag v1_20261019-1200
"""
import argparse
import csv
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

plt.rcParams.update({"font.size": 8, "font.family": "sans-serif", "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 200})
SPLITS = ["id_val", "ood_val", "ood_test"]
HOSP = {0: "H0 (train)", 3: "H3 (train)", 4: "H4 (train)", 1: "H1 (OOD val)", 2: "H2 (OOD test)"}


def auroc(pos, neg):
    s = np.concatenate([pos, neg]); r = s.argsort().argsort() + 1
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def save(fig, out, name, tag):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}_{tag}.{ext}"), bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tag", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for d in sorted(glob.glob(os.path.join(a.runs, "cam_*"))):
        f = os.path.join(d, "result.json")
        if not os.path.exists(f) or "smoke" in d:
            continue
        r = json.load(open(f)); i = r["info"]
        rows.append(dict(run=os.path.basename(d), method=i["adapter"], lr=i["lr"], rank=i.get("rank"),
                         trainable=i["trainable"], epoch=r["selected_epoch"], **{s: r[s] for s in SPLITS},
                         overrides=r.get("overrides")))
    best = {}
    for r in rows:
        if r["method"] not in best or r["id_val"] > best[r["method"]]["id_val"]:
            best[r["method"]] = r
    with open(os.path.join(a.out, f"camelyon_results_{a.tag}.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "method", "lr", "rank", "trainable", "sel_epoch", *SPLITS, "selected_on_id_val"])
        for r in sorted(rows, key=lambda r: (r["method"], r["lr"])):
            w.writerow([r["run"], r["method"], r["lr"], r["rank"], r["trainable"], r["epoch"],
                        *[f"{r[s]:.4f}" for s in SPLITS], int(best[r["method"]] is r)])
    summary = {m: {k: r[k] for k in ("run", "lr", "trainable", *SPLITS)} for m, r in best.items()}

    if "lena" in best and best["lena"]["overrides"]:
        ov = best["lena"]["overrides"]; conds = list(ov)
        fig, ax = plt.subplots(figsize=(3.4, 2.2))
        x = np.arange(len(SPLITS)); wdt = 0.8 / len(conds)
        for k, c in enumerate(conds):
            ax.bar(x + k * wdt, [ov[c][s] for s in SPLITS], wdt, label=c)
        if "lora" in best:
            for j, s in enumerate(SPLITS):
                ax.plot([j - 0.1, j + 0.8], [best["lora"][s]] * 2, "k--", lw=0.8, label="LoRA (selected)" if j == 0 else None)
        ax.set_xticks(x + 0.4 - wdt / 2); ax.set_xticklabels(["ID val", "OOD val (H1)", "OOD test (H2)"])
        ax.set_ylim(0.5, 1.0); ax.set_ylabel("accuracy"); ax.legend(frameon=False, fontsize=6, ncol=2)
        ax.set_title("Camelyon17: LeNA with the fit switched off (s=0) or unshrunk (s=1)", fontsize=7)
        save(fig, a.out, "C_overrides", a.tag)
        summary["lena_overrides"] = ov

        st = os.path.join(a.runs, best["lena"]["run"], "best_stats.pt")
        if os.path.exists(st):
            S = torch.load(st, map_location="cpu", weights_only=False)
            per_img, cen = [], []
            for sp in SPLITS:
                if S.get(sp) is None:
                    continue
                s_ = S[sp]["s"].float(); F_ = torch.nan_to_num(S[sp]["F"].float(), posinf=65504.0)
                per_img.append(dict(s=(s_ > 0).float().mean((1, 2)).numpy(), F=np.log1p(F_.clamp_min(0)).mean((1, 2)).numpy(),
                                    c=S[sp]["center"].numpy(), ok=S[sp]["correct"].numpy()))
            s_all = np.concatenate([d["s"] for d in per_img]); F_all = np.concatenate([d["F"] for d in per_img])
            c_all = np.concatenate([d["c"] for d in per_img])
            fig, ax = plt.subplots(1, 2, figsize=(7.1, 2.2))
            for h in (0, 3, 4, 1, 2):
                if (c_all == h).any():
                    ax[0].hist(s_all[c_all == h], 40, density=True, histtype="step", lw=1.2, label=HOSP[h])
                    ax[1].hist(F_all[c_all == h], 40, density=True, histtype="step", lw=1.2, label=HOSP[h])
            ax[0].set_xlabel("fraction of channels with s>0 (per image)"); ax[1].set_xlabel("mean log(1+F) (per image)")
            ax[0].legend(frameon=False, fontsize=6)
            idm = np.isin(c_all, [0, 3, 4])
            au = {h: float(auroc(F_all[c_all == h], F_all[idm])) for h in (1, 2) if (c_all == h).any()}
            ax[1].set_title("AUROC vs train hospitals: " + ", ".join(f"H{h} {max(v, 1 - v):.2f}" for h, v in au.items()),
                            fontsize=7)
            fig.suptitle("Camelyon17: how much each hospital's images use the per-image fit", fontsize=8, y=1.03)
            save(fig, a.out, "C_hospital_fit", a.tag)
            summary["shift_auroc_logF"] = au
            summary["frac_s_pos_by_hospital"] = {HOSP[h]: float(s_all[c_all == h].mean()) for h in (0, 3, 4, 1, 2)
                                                 if (c_all == h).any()}
    json.dump(dict(rows=[{k: v for k, v in r.items() if k != "overrides"} for r in rows], selected=summary),
              open(os.path.join(a.out, f"camelyon_summary_{a.tag}.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
