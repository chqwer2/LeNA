"""How far the per-input fitted LeNA curves bend away from the shared prior (MRI runs, curves.npz).

theta_dev: (visits/slices, layers, r, K) = theta - rho on the knot grid linspace(-3, 3, 16) (spacing 0.4).
Per run and layer: median |theta-rho| over all knots, p90, median over curves of the max knot deviation,
median Stein s. Row layer = 'all' pools the layers.

  python diag_curve_bend.py out.csv runs/mri_s0_lena runs/mri_diag_lena_lr2e-4
"""
import csv
import sys

import numpy as np

out, runs = sys.argv[1], sys.argv[2:]
rows = []
for d in runs:
    z = np.load(f"{d}/curves.npz")
    t, s = np.abs(z["theta_dev"]), z["stat_s"]
    for li in ["all"] + list(range(t.shape[1])):
        tt = t if li == "all" else t[:, li]
        ss = s if li == "all" else s[:, li]
        rows.append(dict(run=d.split("/")[-1], layer=li, n_inputs=t.shape[0],
                         med_abs_dev=round(float(np.median(tt)), 4), p90_abs_dev=round(float(np.percentile(tt, 90)), 4),
                         med_max_knot_dev=round(float(np.median(tt.max(-1))), 4), med_s=round(float(np.median(ss)), 4),
                         knot_spacing=0.4))
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(open(out).read())
