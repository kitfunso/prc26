"""Q2: a label-free copy detector for LIRF departures (copy = recorded off-block equals the schedule, y = zs).
Features visible on a rank row: airline prefix, flight number, number shape, stand, local hours, zs, local-date gap,
arrival-copy share around takeoff, inbound same-stand arrival copy. Target encodings and the logistic fit use only
training months. Dev rows are scored out of fold over the train.py month pairs; holdout and rank rows by a fit on
all ten dev months. Run: PYTHONPATH=. python analysis/hunt/h1a/copymodel.py  (writes copyprob_lirf.parquet)"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from burst import arrival_copy_share
from common import OUT, movements, v34_dev, v34_holdout, v34_rank
from inbound import attach_inbound
from lirf_b import fmt
from origin import local_day

FOLDS = ((2, 3), (4, 5), (6, 8), (9, 10), (11, 12))
FMTS = ("plain", "dbl1", "dbl2", "none", "other")
ZMIN = 1800


def lirf_frame() -> pd.DataFrame:
    """Every LIRF departure (dev, holdout, rank) with label-free features and v34's prediction p."""
    df = movements()
    df = df[df["apt"] == "LIRF"].copy()
    arr = df[(df["PHASE_mvt"] == "ARR") & df["BLOCK_TIME_UTC_mvt"].notna()]
    acopy = ((arr["BLOCK_TIME_UTC_mvt"] - arr["SCHED_TIME_UTC_mvt"]).abs() <= 10).values.astype(float)
    nost = df[(df["PHASE_mvt"] == "DEP") & df["STAND_mvt"].isna()]
    d = pd.concat([attach_inbound(df), nost], ignore_index=True)
    for w in (1800, 3600, 7200):
        d[f"acs{w // 60}"], d[f"acn{w // 60}"] = arrival_copy_share(
            d["MVT_TIME_UTC_mvt"].values, arr["BLOCK_TIME_UTC_mvt"].values, acopy, w)
    d["lane"] = np.where(d["has_plan"], "A", "B")
    d["zs"] = d["MVT_TIME_UTC_mvt"] - d["SCHED_TIME_UTC_mvt"]
    d["fmt"] = d["FLIGHT_mvt"].map(fmt)
    d["pfx"] = d["FLIGHT_mvt"].str[:3].fillna("none")
    d["flt"] = d["FLIGHT_mvt"].fillna("none")
    d["stand1"] = d["STAND_mvt"].str[:1].fillna("none")
    rome = lambda c: pd.to_datetime(d[c], unit="s", utc=True).dt.tz_convert("Europe/Rome").dt.hour
    d["lh_mvt"], d["lh_std"] = rome("MVT_TIME_UTC_mvt"), rome("SCHED_TIME_UTC_mvt")
    d["dday"] = local_day(d["MVT_TIME_UTC_mvt"] - 900, d["apt"]) - local_day(d["SCHED_TIME_UTC_mvt"], d["apt"])
    d["copy"] = ((d["y"] - d["zs"]).abs() <= 10).astype(float).where(d["y"].notna())
    d["part"] = np.where(d["dev"], "dev", np.where(d["ho"], "holdout", "rank"))
    pred = pd.concat([v34_dev()[["id", "p"]], v34_holdout()[["id", "p"]], v34_rank()])
    return d.merge(pred, left_on="MVT_ID_mvt", right_on="id", how="left")


def te(train: pd.DataFrame, apply: pd.DataFrame, col: str, prior: pd.Series, m: float) -> pd.Series:
    """Smoothed copy rate of `col` from train rows, shrunk to the per-row prior of the apply rows."""
    g = train.groupby(col)["copy"].agg(["sum", "count"])
    s = apply[col].map(g["sum"]).fillna(0.0)
    n = apply[col].map(g["count"]).fillna(0.0)
    return (s + m * prior) / (n + m)


def encode(train: pd.DataFrame, apply: pd.DataFrame) -> pd.DataFrame:
    g = pd.Series(train["copy"].mean(), index=apply.index)
    out = pd.DataFrame(index=apply.index)
    out["pfx_te"] = te(train, apply, "pfx", g, 5.0)
    out["flt_te"] = te(train, apply, "flt", out["pfx_te"], 3.0)
    out["stand_te"] = te(train, apply, "STAND_mvt", g, 5.0)
    return out


def design(d: pd.DataFrame, enc: pd.DataFrame) -> np.ndarray:
    lg = lambda p: np.log(p.clip(0.01, 0.99) / (1 - p.clip(0.01, 0.99)))
    x = pd.DataFrame({"pfx": lg(enc["pfx_te"]), "flt": lg(enc["flt_te"]), "stand": lg(enc["stand_te"]),
                      "lzs": np.log(d["zs"].clip(lower=60)), "dday": d["dday"].clip(-1, 2),
                      "night": d["lh_mvt"].between(0, 5).astype(float),
                      "hs": np.sin(2 * np.pi * d["lh_mvt"] / 24), "hc": np.cos(2 * np.pi * d["lh_mvt"] / 24),
                      "acs60": d["acs60"].fillna(0), "acn60": np.log1p(d["acn60"].fillna(0)),
                      "in_copy": d["in_copy"].astype(float).fillna(0), "in_na": d["in_copy"].isna().astype(float)},
                     index=d.index)
    for f in FMTS:
        x[f"fmt_{f}"] = (d["fmt"] == f).astype(float)
    return x.values


def fit_predict(train: pd.DataFrame, apply: pd.DataFrame, c: float = 0.3) -> np.ndarray:
    """Train-side encodings are leave-one-month-out so the logistic weights see honest target encodings."""
    enc_tr = pd.concat([encode(train[train["month"] != mo], train[train["month"] == mo])
                        for mo in train["month"].unique()]).loc[train.index]
    sc = StandardScaler().fit(design(train, enc_tr))
    lr = LogisticRegression(C=c, max_iter=2000).fit(sc.transform(design(train, enc_tr)), train["copy"].values)
    return lr.predict_proba(sc.transform(design(apply, encode(train, apply))))[:, 1]


def score(pop: pd.DataFrame) -> pd.Series:
    """Out-of-fold copy probability for dev rows, and a ten-month fit for holdout and rank rows."""
    pc = pd.Series(np.nan, index=pop.index)
    dev = pop[pop["part"] == "dev"]
    for f in FOLDS:
        te_m = dev["month"].isin(f)
        pc[dev.index[te_m]] = fit_predict(dev[~te_m], dev[te_m])
    rest = pop[pop["part"] != "dev"]
    pc[rest.index] = fit_predict(dev, rest)
    return pc


def main() -> None:
    d = lirf_frame()
    pop = d[(d["PHASE_mvt"] == "DEP") & (d["zs"] > ZMIN)].copy()
    out = []
    for lane in ("B", "A"):
        s = pop[pop["lane"] == lane].copy()
        s["pc"] = score(s)
        out.append(s)
    res = pd.concat(out)
    keep = ["MVT_ID_mvt", "part", "month", "lane", "zs", "dday", "pfx", "flt", "fmt", "STAND_mvt", "lh_mvt", "acs60",
            "in_copy", "copy", "y", "p", "pc", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"]
    res[keep].to_parquet(OUT / "copyprob_lirf.parquet")
    dv = res[res["part"] == "dev"]
    for (lane, zb), s in dv.groupby(["lane", pd.cut(dv["zs"] / 3600, [0.5, 1, 2, 4, 12, 99])], observed=True):
        auc = roc_auc_score(s["copy"], s["pc"]) if s["copy"].nunique() == 2 else np.nan
        print(f"lane {lane} zs {zb}: n={len(s)} copy rate {s['copy'].mean():.3f} mean pc {s['pc'].mean():.3f}"
              f" AUC {auc:.3f} Brier {((s['pc'] - s['copy']) ** 2).mean():.4f} base {s['copy'].var(ddof=0):.4f}")
    print(res.groupby(["lane", "part"]).size().to_string())


if __name__ == "__main__":
    main()
