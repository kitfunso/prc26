"""Lane B at LIRF: P(schedule copy) from a shallow CatBoost on the few fields that move it, times z_sched, plus (1 - P) times the
band's non-copy mean. Every other row keeps the donor. `python laneb.py cv NAME DONOR` / `python laneb.py fit NAME DONOR`."""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from data import CACHE, rmse
from features import build
from train import FOLDS, HOLDOUT, RESULTS, boot_delta, report

APT, COPY, MIN_NONCOPY = "LIRF", 600.0, 20
EDGES = (0.0, 3600.0, 7200.0, 14400.0, 28800.0, 43200.0)
COLS = ["z_sched", "occ", "t_occ_h", "hour", "band", "airline"]
CB = dict(iterations=300, depth=3, learning_rate=0.05, l2_leaf_reg=10.0, random_seed=0, verbose=False, allow_writing_files=False, thread_count=4)
CONFIGS = {"lb_lirf": dict()}


def band(z: np.ndarray) -> np.ndarray:
    return np.searchsorted(np.asarray(EDGES), np.asarray(z, float), "left")


def frame(d: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"z_sched": d["z_sched"].values, "occ": d["occ"].values.astype(float), "t_occ_h": d["t_occ"].fillna(0.0).values / 3600.0,
                         "hour": d["hour"].values.astype(float), "band": band(d["z_sched"].values).astype(float), "airline": d["airline"].astype(str).values})


def rows(d: pd.DataFrame) -> np.ndarray:
    return ((d["has_plan"] == 0) & (d["apt"] == APT)).values


def predict(tr: pd.DataFrame, te: pd.DataFrame, base: np.ndarray) -> np.ndarray:
    """tr, te: LIRF lane-B rows; base: the donor's prediction on te, kept where the band has too few non-copy rows."""
    c = (tr["y"] - tr["z_sched"]).abs().values <= COPY
    m = CatBoostClassifier(**CB).fit(Pool(frame(tr), c.astype(int), cat_features=["airline"]))
    p = m.predict_proba(Pool(frame(te), cat_features=["airline"]))[:, 1]
    tb, zb, out = band(tr["z_sched"].values), band(te["z_sched"].values), base.copy()
    for b in np.unique(zb):
        k = (tb == b) & ~c
        if k.sum() >= MIN_NONCOPY:
            j = zb == b
            out[j] = p[j] * te["z_sched"].values[j] + (1 - p[j]) * tr["y"].values[k].mean()
    return np.maximum(out, 0)


def _apply(d: pd.DataFrame, donor: pd.DataFrame, col: str, fit_rows: pd.DataFrame) -> pd.DataFrame:
    """donor (id, col) with the LIRF lane-B rows of d re-predicted from fit_rows."""
    out = d[["id"]].merge(donor, on="id")
    m = rows(d)
    out.loc[m, col] = predict(fit_rows[rows(fit_rows)], d[m], out.loc[m, col].values)
    return out


def cv(name: str, donor: str) -> None:
    f = build(); f = f[f["split"] == "train"]
    tr, ho = f[~f["month"].isin(HOLDOUT)], f[f["month"].isin(HOLDOUT)]
    do = pd.read_parquet(CACHE / f"{donor}_oof.parquet")
    oof = tr[["id"]].merge(do, on="id")
    for i, months in enumerate(FOLDS):
        va = tr["month"].isin(months).values
        part = _apply(tr[va], do, "oof", tr[~va])
        oof.loc[va, "oof"] = part["oof"].values
        k = rows(tr[va]); yv = tr["y"].values[va]
        print(f"fold {i} {months}: LIRF laneB {rmse(yv[k], part['oof'].values[k]):.1f} (donor {rmse(yv[k], do.set_index('id').loc[tr['id'].values[va][k], 'oof'].values):.1f})", flush=True)
    dh = pd.read_parquet(CACHE / f"{donor}_holdout.parquet")[["id", "p"]]
    h = _apply(ho, dh, "p", tr)
    oof.to_parquet(CACHE / f"{name}_oof.parquet"); h.to_parquet(CACHE / f"{name}_holdout.parquet")
    hb = ho.merge(dh.rename(columns={"p": "base"}), on="id").merge(h, on="id")
    k = rows(hb); d = boot_delta(hb["y"].values, hb["base"].values, hb["p"].values, hb["day"].values)
    line = (f"| {name} (rest from {donor}) | {json.dumps(report(tr['y'].values, oof['oof'].values, tr))} | {json.dumps(report(hb['y'].values, hb['p'].values, hb))} "
            f"| LIRF laneB {rmse(hb['y'].values[k], hb['base'].values[k]):.1f} -> {rmse(hb['y'].values[k], hb['p'].values[k]):.1f} | vs {donor}: {d[0]:+.2f} [{d[1]:+.2f}, {d[2]:+.2f}] |\n")
    RESULTS.open("a", encoding="utf-8").write(line)
    print(line)


def fit(name: str, donor: str) -> None:
    f = build()
    tr, rk = f[f["split"] == "train"], f[f["split"] == "rank"]
    d = pd.read_parquet(f"data/{donor}_rank.parquet").rename(columns={"MVT_ID_mvt": "id", "TAXITIME_SEC_mvt": "p"})
    out = _apply(rk, d, "p", tr).rename(columns={"id": "MVT_ID_mvt", "p": "TAXITIME_SEC_mvt"})
    out.to_parquet(f"data/{name}_rank.parquet")
    print(f"data/{name}_rank.parquet", len(out), float(out["TAXITIME_SEC_mvt"].mean()))


if __name__ == "__main__":
    {"cv": lambda: cv(sys.argv[2], sys.argv[3]), "fit": lambda: fit(sys.argv[2], sys.argv[3])}[sys.argv[1]]()
