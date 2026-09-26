"""Lane A second stage: LightGBM on what the lane A blend leaves (y - p), from the blend, the airport and the distance
from the blend to each plan stamp, so rows whose recorded off-block was copied from a stamp can move toward it
(docs/RECAL.md). `python recal.py cv` / `python recal.py fit`, optionally followed by `member:weight ...` in place of W."""
from __future__ import annotations

import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from data import CACHE, rmse
from regime import slice_delta
from train import FOLDS, HOLDOUT, boot_delta

W = {"ra_cb_wz": 0.36, "ra_lgb_pw_t7": 0.235, "nn3_pw_t7": 0.405}
Z = ["z_aobt", "z_sched", "z_eobt", "z_lobt", "z_iobt"]
COLS = ["p", "apt", *[f"d_{z}" for z in Z]]
GROUPS = (HOLDOUT, *FOLDS)
PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=63, min_child_samples=200, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, verbose=-1, n_jobs=8)


def load() -> pd.DataFrame:
    f = pd.read_parquet(CACHE / "feat.parquet", columns=["id", "split", "month", "day", "apt", "has_plan", "y", *Z])
    f = f[(f["has_plan"] == 1) & f["split"].isin(("train", "rank"))].reset_index(drop=True)
    rk = (f["split"] == "rank").values
    f["p"] = 0.0
    for n, w in W.items():
        tr = pd.concat([pd.read_parquet(CACHE / f"{n}_oof.parquet").rename(columns={"oof": "p"}), pd.read_parquet(CACHE / f"{n}_holdout.parquet")])
        r = pd.read_parquet(f"data/{n}_rank.parquet").set_index("MVT_ID_mvt")["TAXITIME_SEC_mvt"]
        f.loc[~rk, "p"] += w * f.loc[~rk, "id"].map(tr.set_index("id")["p"]).values
        f.loc[rk, "p"] += w * f.loc[rk, "id"].map(r).values
    assert f["p"].notna().all(), "lane A row missing from a blend input"
    f["p"] = np.maximum(f["p"], 0)
    f["apt"] = f["apt"].astype(str).astype("category")
    for z in Z:
        f[f"d_{z}"] = f[z] - f["p"]
    return f


def cv() -> None:
    t = load().query("split == 'train'").reset_index(drop=True)
    grp = t["month"].map({m: i for i, g in enumerate(GROUPS) for m in g}).values
    e, out = (t["y"] - t["p"]).values, np.zeros(len(t))
    for g in range(len(GROUPS)):
        m = grp == g
        out[m] = lgb.LGBMRegressor(**PARAMS).fit(t.loc[~m, COLS], e[~m]).predict(t.loc[m, COLS])
        print(f"group {GROUPS[g]} done", flush=True)
    q, ho = np.maximum(t["p"].values + out, 0), t["month"].isin(HOLDOUT).values
    pd.DataFrame({"id": t["id"].values[~ho], "oof": q[~ho]}).to_parquet(CACHE / "recal_oof.parquet")
    pd.DataFrame({"id": t["id"].values[ho], "p": q[ho]}).to_parquet(CACHE / "recal_holdout.parquet")
    y, p, day = t["y"].values, t["p"].values, t["day"].values
    d, a = boot_delta(y[ho], p[ho], q[ho], day[ho]), boot_delta(y, p, q, day)
    print(f"lane A holdout: blend {rmse(y[ho], p[ho]):.2f}, recal {rmse(y[ho], q[ho]):.2f}, delta {d[0]:+.2f} [{d[1]:+.2f}, {d[2]:+.2f}]"
          f" | 12 months {a[0]:+.2f} [{a[1]:+.2f}, {a[2]:+.2f}]")
    print(f"  regime: {slice_delta(y[ho], p[ho], q[ho], day[ho], t['id'].values[ho])}")


def fit() -> None:
    f = load()
    tr, rk = (f["split"] == "train").values, (f["split"] == "rank").values
    m = lgb.LGBMRegressor(**PARAMS).fit(f.loc[tr, COLS], (f["y"] - f["p"]).values[tr])
    q = np.maximum(f.loc[rk, "p"].values + m.predict(f.loc[rk, COLS]), 0)
    lb = pd.read_parquet("data/laneb_rank.parquet")
    out = pd.concat([pd.DataFrame({"MVT_ID_mvt": f.loc[rk, "id"].values, "TAXITIME_SEC_mvt": q}), lb], ignore_index=True)
    assert out["MVT_ID_mvt"].is_unique, "a lane B row is also a lane A row"
    out.to_parquet("data/recal_rank.parquet")
    print(f"data/recal_rank.parquet {len(out)} rows; lane A {rk.sum()} rows, mean blend {f.loc[rk, 'p'].mean():.1f}, recal {q.mean():.1f}")


if __name__ == "__main__":
    if len(sys.argv) > 2:
        W = {n: float(w) for n, w in (a.rsplit(":", 1) for a in sys.argv[2:])}
    {"cv": cv, "fit": fit}[sys.argv[1]]()
