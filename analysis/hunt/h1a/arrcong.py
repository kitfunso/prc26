"""Q3: is surface congestion that stays visible on rank days (arrival taxi-in, neighbour departures' schedule delay,
runway counts) missing from v34? Level-2 LightGBM on v34's dev OOF residual, out of fold by the train.py month pairs,
with and without the congestion features. Windows end at the departure's takeoff; the row itself is excluded.
Run: PYTHONPATH=. python analysis/hunt/h1a/arrcong.py  (writes arrcong_feat.parquet)"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import OUT, movements, v34_dev, v34_holdout, v34_rank
from copymodel import FOLDS

WIN = (1800, 3600, 7200)
BIG = 43200
BASE = ["apt_c", "laneB", "zs", "p", "hr"]
CONG = [f"{a}{w // 60}" for w in WIN for a in ("an", "am", "as", "dn", "dm", "ds")]


def wstats(tq: np.ndarray, ts: np.ndarray, vs: np.ndarray, win: float, thr: float) -> tuple:
    """Count, mean and share above thr of the values vs whose time ts falls in [tq - win, tq)."""
    o = np.argsort(ts)
    t, v = ts[o], vs[o]
    cs, cg = np.r_[0, np.cumsum(v)], np.r_[0, np.cumsum(v > thr)]
    lo, hi = np.searchsorted(t, tq - win), np.searchsorted(t, tq)
    n = hi - lo
    m = np.maximum(n, 1)
    return n, np.where(n > 0, (cs[hi] - cs[lo]) / m, np.nan), np.where(n > 0, (cg[hi] - cg[lo]) / m, np.nan)


def features(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for apt, g in df.groupby("apt"):
        arr = g[(g["PHASE_mvt"] == "ARR") & g["BLOCK_TIME_UTC_mvt"].notna()]
        tin = arr["BLOCK_TIME_UTC_mvt"] - arr["MVT_TIME_UTC_mvt"]
        ok = ((arr["BLOCK_TIME_UTC_mvt"] - arr["SCHED_TIME_UTC_mvt"]).abs() > 10) & tin.between(0, 10800)
        at, av = arr.loc[ok, "BLOCK_TIME_UTC_mvt"].values, tin[ok].clip(upper=3600).values
        base = float(np.median(av[(arr.loc[ok, "split"] == "train").values]))
        d = g[g["PHASE_mvt"] == "DEP"].copy()
        tq = d["MVT_TIME_UTC_mvt"].values
        dz = (d["MVT_TIME_UTC_mvt"] - d["SCHED_TIME_UTC_mvt"]).clip(-1800, 14400).values
        for w in WIN:
            k = w // 60
            d[f"an{k}"], m, d[f"as{k}"] = wstats(tq, at, av, w, 900)
            d[f"am{k}"] = m - base
            d[f"dn{k}"], d[f"dm{k}"], d[f"ds{k}"] = wstats(tq, tq, dz, w, 3600)
        out.append(d)
    return pd.concat(out)


def frame() -> pd.DataFrame:
    df = movements()[["MVT_ID_mvt", "PHASE_mvt", "apt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt",
                      "SCHED_TIME_UTC_mvt", "has_plan", "month", "split", "dev", "ho", "y"]]
    d = features(df)
    d["zs"] = d["MVT_TIME_UTC_mvt"] - d["SCHED_TIME_UTC_mvt"]
    d["laneB"] = (~d["has_plan"].astype(bool)).astype(int)
    d["apt_c"] = d["apt"].astype("category")
    d["hr"] = (d["MVT_TIME_UTC_mvt"] // 3600) % 24
    d["part"] = np.where(d["dev"], "dev", np.where(d["ho"], "holdout", "rank"))
    pred = pd.concat([v34_dev()[["id", "p"]], v34_holdout()[["id", "p"]], v34_rank()])
    return d.merge(pred, left_on="MVT_ID_mvt", right_on="id", how="inner")


def fit_predict(tr: pd.DataFrame, te: pd.DataFrame, cols: list[str]) -> np.ndarray:
    m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=500,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=10.0, n_jobs=4,
                          verbose=-1)
    m.fit(tr[cols], tr["y"] - tr["p"], categorical_feature=["apt_c"])
    return m.predict(te[cols])


def sse(s: pd.DataFrame, c: str) -> float:
    return float(((s["y"] - s[c]) ** 2).sum() / 1e9)


def main() -> None:
    d = frame()
    d.drop(columns=["apt_c"]).to_parquet(OUT / "arrcong_feat.parquet")
    dv = d[d["part"] == "dev"].copy()
    dv["yb"] = pd.cut(dv["y"], [-1e9, 3600, BIG, 1e12], labels=["<=1h", "1-12h", ">12h"])
    pd.set_option("display.width", 250)
    print(dv.groupby("yb", observed=True)[CONG].mean().round(2).T.to_string())
    for name, cols in (("base", BASE), ("base+cong", BASE + CONG)):
        for f in FOLDS:
            te_m = dv["month"].isin(f)
            tr = dv[~te_m & (dv["y"] <= BIG)]
            dv.loc[te_m, name] = dv.loc[te_m, "p"] + fit_predict(tr, dv[te_m], cols)
        print(f"\n{name}: dev SSE v34 {sse(dv, 'p'):.3f}B level-2 {sse(dv, name):.3f}B")
        for key in ("month", "apt", "laneB", "yb"):
            g = dv.groupby(dv["month"].map({m: str(f) for f in FOLDS for m in f}) if key == "month" else key,
                           observed=True)
            t = pd.DataFrame({"n": g.size(), "v34": g.apply(sse, "p"), "l2": g.apply(sse, name)})
            t["saved_M"] = (t["v34"] - t["l2"]) * 1e3
            print(t.round(3).to_string())


if __name__ == "__main__":
    main()
