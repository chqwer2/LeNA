"""Figures V1-V8 and the E5-E8 numbers from a diagnose_lena.py dump (+ its json).

  python plot_diag.py --dump diag_v1_dump.pt --json diag_v1.json --out figures/commonsense --tag v1_20261019-0130
Writes <name>_<tag>.png/.pdf per figure and numbers_<tag>.json.
"""
import argparse
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

TASKS = ["boolq", "piqa", "social_i_qa", "hellaswag", "winogrande", "ARC-Challenge", "ARC-Easy", "openbookqa"]
SHORT = ["BoolQ", "PIQA", "SIQA", "HellaS.", "WinoG.", "ARC-c", "ARC-e", "OBQA"]
COL = plt.get_cmap("tab10").colors
plt.rcParams.update({"font.size": 8, "font.family": "sans-serif", "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 200})


def mod_index(names):
    """(layer, type) per module name, e.g. model.layers.12.mlp.up_proj -> (12, 'up')."""
    out = []
    for n in names:
        m = re.search(r"layers\.(\d+)\.\w+\.(\w+)_proj", n)
        out.append((int(m.group(1)), m.group(2)))
    return out


def save(fig, out, name, tag):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}_{tag}.{ext}"), bbox_inches="tight")
    plt.close(fig)


def grid(values, idx, types):
    L = max(l for l, _ in idx) + 1
    G = np.full((len(types), L), np.nan)
    for v, (l, t) in zip(values, idx):
        G[types.index(t), l] = v
    return G


def auroc(pos, neg):
    """P(score_pos > score_neg) (Mann-Whitney)."""
    s = np.concatenate([pos, neg]); r = s.argsort().argsort() + 1
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def knn_task_acc(X, y, k=5):
    """Leave-one-out k-NN task accuracy from fitted curves (cosine distance)."""
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    S = Xn @ Xn.T; np.fill_diagonal(S, -np.inf)
    nb = np.argsort(-S, 1)[:, :k]
    pred = np.array([np.bincount(y[r], minlength=y.max() + 1).argmax() for r in nb])
    return (pred == y).mean()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dump", required=True)
    p.add_argument("--json", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tag", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    D = torch.load(a.dump, map_location="cpu", weights_only=False)
    J = json.load(open(a.json))
    idx = mod_index(D["names"]); types = [t for t in ("q", "k", "v", "up", "down") if any(t == x for _, x in idx)]
    task = np.array([TASKS.index(t) for t in D["tasks"]])
    s, Fs, theta, rho = D["s"].float().numpy(), D["F"].float().numpy(), D["theta"].float().numpy(), D["rho"].numpy()
    E, M, r, K = theta.shape
    num = {}

    # E5: is the step used? fraction of (input, module, channel) with s > 0, and with s == 1-ish
    num["E5_frac_s_pos"] = float((s > 0).mean()); num["E5_mean_s"] = float(s.mean())
    num["E5_frac_s_gt_0.9"] = float((s > 0.9).mean())
    # V4: where map
    fpos = (s > 0).mean((0, 2)); dev = np.sqrt(((rho - np.linspace(-3, 3, K)) ** 2).mean((1, 2)))
    fig, ax = plt.subplots(2, 1, figsize=(7.1, 2.6), sharex=True)
    for axi, G, lab, cm in ((ax[0], grid(fpos, idx, types), "fraction s>0", "viridis"),
                            (ax[1], grid(dev, idx, types), "RMS(ρ − identity)", "magma")):
        im = axi.imshow(G, aspect="auto", cmap=cm); axi.set_yticks(range(len(types))); axi.set_yticklabels(types)
        plt.colorbar(im, ax=axi, fraction=0.02, pad=0.01, label=lab)
    ax[1].set_xlabel("layer"); ax[0].set_title("V4  where the per-input fit engages (top) and where the prior bends (bottom)")
    save(fig, a.out, "V4_where_map", a.tag)

    # V1 + V3: fitted curves identify the task; their spread is higher-rank than one LoRA
    X = (theta - theta.mean(0, keepdims=True)).reshape(E, M, r * K)
    knn = np.array([knn_task_acc(theta[:, m].reshape(E, -1), task) for m in range(M)])
    pr = []
    for m in range(M):
        sv = np.linalg.svd(X[:, m], compute_uv=False) ** 2
        pr.append(sv.sum() ** 2 / (sv ** 2).sum() if sv.sum() > 0 else 0.0)
    pr = np.array(pr)
    num["V1_knn_task_acc_best"] = float(knn.max()); num["V1_knn_task_acc_mean"] = float(knn.mean())
    num["V1_chance"] = float(np.bincount(task).max() / E)
    num["V3_participation_ratio_mean"] = float(pr.mean()); num["V3_participation_ratio_max"] = float(pr.max())
    mb = int(knn.argmax())
    U, S_, Vt = np.linalg.svd(X[:, mb], full_matrices=False); Z = U[:, :2] * S_[:2]
    fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.3), layout="constrained")
    for t in range(len(TASKS)):
        sel = task == t
        ax[0].scatter(Z[sel, 0], Z[sel, 1], s=4, color=COL[t % 10], label=SHORT[t], alpha=0.7, lw=0)
    ax[0].set_title(f"V1  curves of {D['names'][mb].split('model.')[-1]}", fontsize=7)
    ax[0].set_xlabel("PC1"); ax[0].set_ylabel("PC2"); ax[0].legend(fontsize=5, markerscale=2, frameon=False, ncol=2)
    im = ax[1].imshow(grid(knn, idx, types), aspect="auto", cmap="viridis", vmin=num["V1_chance"], vmax=1)
    ax[1].set_yticks(range(len(types))); ax[1].set_yticklabels(types); ax[1].set_xlabel("layer")
    ax[1].set_title("5-NN task accuracy from curves", fontsize=7); plt.colorbar(im, ax=ax[1], fraction=0.05)
    for m in np.argsort(-pr)[:4]:
        sv = np.linalg.svd(X[:, m], compute_uv=False); ax[2].semilogy(sv / sv[0], lw=1, label=f"{idx[m][1]}{idx[m][0]}")
    ax[2].set_xlim(0, 40); ax[2].set_xlabel("component"); ax[2].set_ylabel("σ / σ₁")
    ax[2].set_title(f"V3  spread of per-input maps (LoRA: none)\nmean participation ratio {pr.mean():.1f}", fontsize=7)
    ax[2].legend(fontsize=5, frameon=False)
    save(fig, a.out, "V1_V3_curves_by_task", a.tag)

    # V2: donor task x target task
    if "swap_matrix" in J:
        Mx = np.array(J["swap_matrix"]["acc"])
        fig, ax = plt.subplots(figsize=(3.4, 3.0))
        im = ax.imshow(Mx, cmap="RdYlGn"); plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_xticks(range(8)); ax.set_xticklabels(SHORT, rotation=60); ax.set_yticks(range(8)); ax.set_yticklabels(SHORT)
        ax.set_xlabel("target task (scored)"); ax.set_ylabel("donor task (statistics)")
        for i in range(8):
            for j in range(8):
                ax.text(j, i, f"{100 * Mx[i, j]:.0f}", ha="center", va="center", fontsize=5)
        ax.set_title("V2  curve fitted from another input", fontsize=8)
        save(fig, a.out, "V2_swap_matrix", a.tag)
        num["V2_diag_mean"] = float(np.diag(Mx).mean()); num["V2_offdiag_mean"] = float(Mx[~np.eye(8, dtype=bool)].mean())

    # V5: shrinkage geometry, s against F per (input, module, channel)
    rng = np.random.RandomState(0); sel = rng.choice(s.size, min(40000, s.size), replace=False)
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.scatter(Fs.ravel()[sel], s.ravel()[sel], s=1, alpha=0.2, color="k", lw=0)
    f = np.logspace(-1, 3, 200)
    for g, ls in ((0.5, ":"), (1.0, "-")):
        ax.plot(f, np.clip(1 - g / f, 0, None), ls, color="C3", lw=1, label=f"(1 − {g}/F)₊")
    ax.set_xscale("log"); ax.set_xlabel("fit statistic F"); ax.set_ylabel("step s"); ax.legend(frameon=False, fontsize=6)
    ax.set_title("V5  keep the prior when evidence is weak", fontsize=8)
    save(fig, a.out, "V5_shrinkage_geometry", a.tag)

    # E6: evidence accumulates along the prefix
    ph = D["pos_hist"]; b = ph["bins"]; fp = ph["frac_pos"].numpy()
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    mids = [(b[i] + min(b[i + 1], 300)) / 2 for i in range(len(b) - 1)]
    for t in types:
        ms = [m for m, (_, x) in enumerate(idx) if x == t]
        ax.plot(mids, fp[ms].mean(0), marker="o", ms=2, lw=1, label=t)
    ax.set_xlabel("prefix length (tokens)"); ax.set_ylabel("fraction s>0"); ax.legend(frameon=False, fontsize=6)
    ax.set_title("E6  the fit switches on as the prefix grows", fontsize=8)
    save(fig, a.out, "E6_prefix_evidence", a.tag)
    num["E6_frac_pos_by_bin"] = fp.mean(0).tolist()

    # V8: shift score, GSM8K prompts vs commonsense prompts (mean log F over modules and channels)
    if "shift_gsm8k" in D:
        sc = lambda F_: np.log1p(np.clip(torch.nan_to_num(F_.float(), posinf=65504.0).numpy(), 0, None)).mean((1, 2))  # noqa: E731
        cin, cout = sc(D["F"]), sc(D["shift_gsm8k"]["F"])
        au = auroc(cout, cin); num["E8_auroc_gsm8k_vs_commonsense"] = float(max(au, 1 - au)); num["E8_direction"] = "higher on GSM8K" if au >= .5 else "lower on GSM8K"
        fig, ax = plt.subplots(figsize=(3.4, 2.2))
        ax.hist(cin, 40, alpha=0.6, density=True, label="commonsense (train dist.)")
        ax.hist(cout, 40, alpha=0.6, density=True, label="GSM8K (unseen)")
        ax.set_xlabel("mean log(1+F) at the last prompt token"); ax.legend(frameon=False, fontsize=6)
        ax.set_title(f"V8  fit statistic as a shift score, AUROC {num['E8_auroc_gsm8k_vs_commonsense']:.2f}", fontsize=8)
        save(fig, a.out, "V8_shift_score", a.tag)

    # V7: one channel, several tasks: (u, v) scatter, prior, fitted curve
    if D.get("scatter"):
        sc0 = D["scatter"]; mi = 0; ch = int(np.argmax(s[:, sc0[0]["mods"][mi]].mean(0)))
        grid_u = np.linspace(-3, 3, K)
        fig, ax = plt.subplots(1, min(4, len(sc0)), figsize=(7.1, 1.9), sharey=True)
        ax = np.atleast_1d(ax)
        for k, d in enumerate(sc0[::max(1, len(sc0) // 4)][:len(ax)]):
            u, v = d["u"][mi, :, ch].float().numpy(), d["v"][mi, :, ch].float().numpy()
            ax[k].scatter(u, v, s=2, color="0.6", lw=0)
            ax[k].plot(grid_u, rho[d["mods"][mi], ch], "--", color="k", lw=1, label="prior ρ")
            ax[k].plot(grid_u, d["theta"][mi, ch].numpy(), color="C3", lw=1.2, label="fitted")
            ax[k].set_title(d["task"], fontsize=7); ax[k].set_xlabel("u = A₁x")
        ax[0].set_ylabel("v = A₂x"); ax[0].legend(frameon=False, fontsize=5)
        fig.suptitle(f"V7  one channel ({D['names'][sc0[0]['mods'][mi]].split('model.')[-1]}, #{ch}) across inputs",
                     fontsize=8, y=1.05)
        save(fig, a.out, "V7_channel_curves", a.tag)

    # element checks from the scoring accuracies
    acc = J["scoring_acc"]
    num["scoring_avg"] = {c: acc[c]["avg"] for c in acc}
    json.dump(num, open(os.path.join(a.out, f"numbers_{a.tag}.json"), "w"), indent=1)
    print(json.dumps(num, indent=1))


if __name__ == "__main__":
    main()
