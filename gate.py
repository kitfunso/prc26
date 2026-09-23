"""File-level holdout gate for a lane A change (docs/EXPERIMENT-PROTOCOL.md section 3): a new second-stage holdout cache
against a base one, lane B rows from laneb_holdout in both arms. `python gate.py [NEW] [BASE]`, cache file names,
default recal_holdout vs recal_holdout_v34 (file of record). Check: the same file in both arms prints +0.000 everywhere."""
import sys

import numpy as np
import pandas as pd

from data import CACHE, rmse
from regime import slice_delta
from train import HOLDOUT, boot_delta

new = sys.argv[1] if len(sys.argv) > 1 else "recal_holdout.parquet"
base = sys.argv[2] if len(sys.argv) > 2 else "recal_holdout_v34.parquet"
f = pd.read_parquet(CACHE / "feat.parquet", columns=["id", "split", "month", "day", "has_plan", "y"])
f = f[(f["split"] == "train") & f["month"].isin(HOLDOUT)].reset_index(drop=True)
lb = pd.read_parquet(CACHE / "laneb_holdout.parquet").set_index("id")["p"]
a = f["has_plan"].values == 1
arms = {}
for k, path in (("base", base), ("new", new)):
    la = pd.read_parquet(CACHE / path).set_index("id")["p"]
    p = np.where(a, f["id"].map(la).values, f["id"].map(lb).values).astype(float)
    assert not np.isnan(p).any(), f"{path}: holdout row without a prediction"
    arms[k] = p
    print(f"{k:4s} {path}: A {rmse(f['y'].values[a], p[a]):.2f} / B {rmse(f['y'].values[~a], p[~a]):.2f} / all {rmse(f['y'].values, p):.2f}")
y, day, ids = f["y"].values, f["day"].values, f["id"].values
d = boot_delta(y, arms["base"], arms["new"], day)
da = boot_delta(y[a], arms["base"][a], arms["new"][a], day[a])
sse = ((y - arms["base"]) ** 2).sum() - ((y - arms["new"]) ** 2).sum()
print(f"all rows {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}] | lane A {da[0]:+.3f} [{da[1]:+.3f}, {da[2]:+.3f}] | SSE gain {sse / 1e6:+.1f}M")
print("regime, lane A:", slice_delta(y[a], arms["base"][a], arms["new"][a], day[a], ids[a]))
