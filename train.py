"""CV, holdout and full refit for one named configuration. `python train.py cv NAME` / `python train.py fit NAME`."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from data import CACHE, rmse
from features import CAT, NUM, build

HOLDOUT = (1, 7)
FOLDS = ((2, 3), (4, 5), (6, 8), (9, 10), (11, 12))
BASE = dict(loss_function="RMSE", task_type="GPU", depth=8, learning_rate=0.1, l2_leaf_reg=3, iterations=6000,
            od_type="Iter", od_wait=200, random_seed=0, verbose=0, border_count=254)
CONFIGS: dict[str, dict] = {
    "cb0": dict(params=BASE, cat=CAT, num=NUM),
}
RESULTS = Path("docs/RESULTS.md")


def boot_delta(y: np.ndarray, p0: np.ndarray, p1: np.ndarray, day: np.ndarray, n: int = 400, seed: int = 0) -> tuple[float, float, float]:
    """Paired day-block bootstrap of RMSE(p0) - RMSE(p1): positive means p1 is better. Returns point, lo, hi."""
    rng = np.random.default_rng(seed)
    days, inv = np.unique(day, return_inverse=True)
    s0 = np.bincount(inv, (y - p0) ** 2); s1 = np.bincount(inv, (y - p1) ** 2); cnt = np.bincount(inv)
    d = []
    for _ in range(n):
        w = rng.multinomial(len(days), np.full(len(days), 1 / len(days)))
        d.append(np.sqrt((w * s0).sum() / (w * cnt).sum()) - np.sqrt((w * s1).sum() / (w * cnt).sum()))
    return float(rmse(y, p0) - rmse(y, p1)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def fit_predict(cfg: dict, tr: pd.DataFrame, te: pd.DataFrame, va: pd.DataFrame | None = None, iters: int | None = None) -> tuple[np.ndarray, int]:
    cols = cfg["cat"] + cfg["num"]
    params = dict(cfg["params"])
    if iters:
        params.update(iterations=iters, od_type=None, od_wait=None)
        params = {k: v for k, v in params.items() if v is not None}
    m = CatBoostRegressor(**params)
    pool = Pool(tr[cols], tr["y"], cat_features=cfg["cat"])
    m.fit(pool, eval_set=Pool(va[cols], va["y"], cat_features=cfg["cat"]) if va is not None else None, use_best_model=va is not None)
    return m.predict(Pool(te[cols], cat_features=cfg["cat"])), m.get_best_iteration() or params["iterations"]


def report(y: pd.Series, p: np.ndarray, f: pd.DataFrame) -> dict[str, float]:
    a = f["has_plan"].values == 1
    out = {"all": rmse(y, p), "laneA": rmse(y[a], p[a]), "laneB": rmse(y[~a], p[~a])}
    for m in HOLDOUT:
        k = f["month"].values == m
        out[f"m{m}"] = rmse(y[k], p[k])
    return {k: round(v, 2) for k, v in out.items()}


def cv(name: str) -> None:
    cfg = CONFIGS[name]
    f = build()
    tr = f[(f["split"] == "train") & ~f["month"].isin(HOLDOUT)]
    ho = f[(f["split"] == "train") & f["month"].isin(HOLDOUT)]
    oof, iters, t0 = np.full(len(tr), np.nan), [], time.time()
    for i, months in enumerate(FOLDS):
        va = tr["month"].isin(months).values
        p, it = fit_predict(cfg, tr[~va], tr[va], tr[va])
        oof[va] = p; iters.append(it)
        print(f"fold {i} {months}: {rmse(tr['y'].values[va], p):.2f} at {it} iters, {time.time() - t0:.0f}s", flush=True)
    cv_all = report(tr["y"].values, oof, tr)
    it = int(np.mean(iters) * 1.1)
    ph, _ = fit_predict(cfg, tr, ho, iters=it)
    ph = np.maximum(ph, 0)
    ho_all = report(ho["y"].values, ph, ho)
    pd.DataFrame({"id": tr["id"].values, "oof": oof}).to_parquet(CACHE / f"{name}_oof.parquet")
    pd.DataFrame({"id": ho["id"].values, "p": ph}).to_parquet(CACHE / f"{name}_holdout.parquet")
    line = f"| {name} | {json.dumps(cv_all)} | {json.dumps(ho_all)} | {it} | {time.time() - t0:.0f}s |\n"
    if not RESULTS.exists():
        RESULTS.write_text("| config | CV (10 months, OOF) | holdout Jan+Jul 2025 (floored) | refit iters | time |\n|---|---|---|---|---|\n", encoding="utf-8")
    RESULTS.open("a", encoding="utf-8").write(line)
    print(line)


def fit(name: str, out: str, iters: int) -> None:
    """Refit on all twelve months (iteration count from the CV line) and write predictions for the ranking rows."""
    cfg = CONFIGS[name]
    f = build()
    tr, rk = f[f["split"] == "train"], f[f["split"] == "rank"]
    p, _ = fit_predict(cfg, tr, rk, iters=iters)
    pd.DataFrame({"MVT_ID_mvt": rk["id"].values, "TAXITIME_SEC_mvt": np.maximum(p, 0)}).to_parquet(out)
    print(out, len(p), float(np.mean(p)))


if __name__ == "__main__":
    {"cv": lambda: cv(sys.argv[2]), "fit": lambda: fit(sys.argv[2], f"data/{sys.argv[2]}_rank.parquet", int(sys.argv[3]))}[sys.argv[1]]()
