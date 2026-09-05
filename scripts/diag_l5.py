"""Diagnostic, no verdict: on the TRAINING month-pair folds (never the holdout), where do L5a and L4b differ by band at LIRF, and what are band 6's non-copy rows?"""
import numpy as np
import pandas as pd

from taxi.data import CV_FOLDS, TRAIN_MONTHS, load
from taxi.l2 import floored
from taxi.l4 import band, touched
from taxi.l5 import CELLS, champion, is_copy, per_row

tr_all = load(TRAIN_MONTHS)
trb = tr_all[~tr_all.has_fpl].reset_index(drop=True)
rows = []
for fold in CV_FOLDS:
    m = trb.month.isin(fold).values
    tr, te = trb[~m].reset_index(drop=True), trb[m].reset_index(drop=True)
    t = touched(tr, te)
    e0, e1 = floored(champion(tr, te)) - te.y.values, floored(per_row(tr, te, CELLS["L5a"])) - te.y.values
    zb = band(te.z_sched.values)
    for b in range(7):
        k = t & (zb == b)
        if k.any():
            rows.append((fold, b, int(k.sum()), float(e0[k] @ e0[k]), float(e1[k] @ e1[k])))
d = pd.DataFrame(rows, columns=["fold", "band", "n", "sse_l4b", "sse_l5a"])
g = d.groupby("band")[["n", "sse_l4b", "sse_l5a"]].sum()
g["rmse_l4b"], g["rmse_l5a"] = np.sqrt(g.sse_l4b / g.n), np.sqrt(g.sse_l5a / g.n)
g["delta_sse"] = g.sse_l5a - g.sse_l4b
print("CV folds pooled, LIRF touched rows by band (L4b vs L5a, both floored):")
print(g[["n", "rmse_l4b", "rmse_l5a", "delta_sse"]].round(0).to_string())
print(f"\ntotal delta SSE {g.delta_sse.sum():.3e}; bands 0-3 only {g.delta_sse.loc[:3].sum():.3e}; band 6 {g.delta_sse.loc[6]:.3e}")
# What are the non-copy rows in each band at LIRF (training rows)? y - z_sched in days.
t = trb[(trb.apt == "LIRF") & (trb.notyp == 1)]
c, zb = is_copy(t), band(t.z_sched.values)
dd = (t.y.values - t.z_sched.values) / 86400
print("\nLIRF no-type training rows by band: n, copy share, non-copy rows' (y - z_sched)/86400 quantiles, non-copy y mean")
for b in range(7):
    k, kn = zb == b, (zb == b) & ~c
    q = np.quantile(dd[kn], [0.1, 0.5, 0.9]) if kn.any() else [np.nan] * 3
    print(f"band {b}: n {k.sum():4d} copy {c[k].mean():.2f} non-copy n {kn.sum():4d} (y-z)/day p10 {q[0]:7.2f} p50 {q[1]:7.2f} p90 {q[2]:7.2f} "
          f"non-copy y mean {t.y.values[kn].mean() if kn.any() else float('nan'):9.0f}")
