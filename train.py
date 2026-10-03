"""CV, holdout and full refit for one named configuration. `python train.py cv NAME` / `python train.py fit NAME`."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, Pool
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data import CACHE, rmse
from features import CAT, DAY, NUM, build

HOLDOUT = (1, 7)
FOLDS = ((2, 3), (4, 5), (6, 8), (9, 10), (11, 12))
BASE = dict(loss_function="RMSE", task_type="GPU", depth=8, learning_rate=0.1, l2_leaf_reg=3, iterations=6000,
            od_type="Iter", od_wait=200, random_seed=0, verbose=0, border_count=254)
LGB = dict(objective="regression", learning_rate=0.05, num_leaves=255, min_child_samples=50, feature_fraction=0.8,
           bagging_fraction=0.8, bagging_freq=1, lambda_l2=3.0, max_bin=255, n_estimators=8000, verbose=-1, random_state=0, n_jobs=20)
XGB = dict(objective="reg:squarederror", learning_rate=0.05, max_depth=9, min_child_weight=20, subsample=0.8, colsample_bytree=0.8,
           reg_lambda=3.0, max_bin=256, n_estimators=8000, tree_method="hist", device="cuda", enable_categorical=True,
           max_cat_to_onehot=1, random_state=0)
CONFIGS: dict[str, dict] = {
    "cb0": dict(params=BASE, cat=CAT, num=NUM),
    "cb1": dict(params=BASE, cat=CAT, num=NUM + DAY),
    "lgb0": dict(family="lgb", params=LGB, cat=CAT, num=NUM),
    "lgb1": dict(family="lgb", params=LGB, cat=CAT, num=NUM + DAY),
    "xgb0": dict(family="xgb", params=XGB, cat=CAT, num=NUM),
}
RESULTS = Path("docs/RESULTS.md")


def stopping_split(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    days = np.sort(rows["day"].unique())
    if len(days) < 2:
        raise ValueError("stopping selection needs at least two training days")
    stop = rows["day"].isin(days[-max(1, int(np.ceil(len(days) * 0.1))):])
    return rows[~stop], rows[stop]


def outer_predict(cfg: dict, tr: pd.DataFrame, te: pd.DataFrame) -> tuple[np.ndarray, int]:
    inner, stop = stopping_split(tr)
    _, iterations = fit_predict(cfg, inner, stop, stop)
    return fit_predict(cfg, tr, te, iters=iterations)


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


def save_model(model: object, path: Path | None, family: str) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if family == "lgb":
        model.booster_.save_model(str(path))
    elif family in ("cb", "xgb"):
        model.save_model(str(path))
    else:
        raise ValueError(f"model persistence is unsupported for {family}")


def fit_predict(cfg: dict, tr: pd.DataFrame, te: pd.DataFrame, va: pd.DataFrame | None = None, iters: int | None = None,
                model_path: Path | None = None) -> tuple[np.ndarray, int]:
    cols = cfg["cat"] + cfg["num"]
    params = dict(cfg["params"])
    w = cfg["weight"](tr) if "weight" in cfg else None
    fam = cfg.get("family", "cb")
    if fam == "lgb":
        n = iters or params.pop("n_estimators")
        m = lgb.LGBMRegressor(**{**params, "n_estimators": n})
        m.fit(tr[cols], tr["y"], sample_weight=w, eval_set=[(va[cols], va["y"])] if va is not None else None,
              callbacks=[lgb.early_stopping(200, verbose=False)] if va is not None else None, categorical_feature=cfg["cat"])
        save_model(m, model_path, fam)
        return m.predict(te[cols]), (m.best_iteration_ or n) if va is not None else n
    if fam == "xgb":
        n = iters or params.pop("n_estimators")
        m = xgb.XGBRegressor(**{**params, "n_estimators": n}, early_stopping_rounds=200 if va is not None else None)
        m.fit(tr[cols], tr["y"], sample_weight=w, eval_set=[(va[cols], va["y"])] if va is not None else None, verbose=False)
        save_model(m, model_path, fam)
        return m.predict(te[cols]), (m.best_iteration + 1) if va is not None else n
    if fam == "ridge":
        pre = ColumnTransformer(
            [("n", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), cfg["num"]),
             ("c", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=params.pop("min_frequency", 500)), cfg["cat"])],
            sparse_threshold=1.0)
        m = Pipeline([("p", pre), ("r", Ridge(**params))])
        m.fit(tr[cols].assign(**{c: tr[c].astype(str) for c in cfg["cat"]}), tr["y"], r__sample_weight=w)
        return m.predict(te[cols].assign(**{c: te[c].astype(str) for c in cfg["cat"]})), 1
    if iters:
        params.update(iterations=iters, od_type=None, od_wait=None)
        params = {k: v for k, v in params.items() if v is not None}
    m = CatBoostRegressor(**params)
    pool = Pool(tr[cols], tr["y"], cat_features=cfg["cat"], weight=w)
    m.fit(pool, eval_set=Pool(va[cols], va["y"], cat_features=cfg["cat"]) if va is not None else None, use_best_model=va is not None)
    save_model(m, model_path, "cb")
    best = m.get_best_iteration()
    return m.predict(Pool(te[cols], cat_features=cfg["cat"])), best + 1 if best >= 0 else params["iterations"]


def report(y: pd.Series, p: np.ndarray, f: pd.DataFrame) -> dict[str, float]:
    a = f["has_plan"].values == 1
    out = {"all": rmse(y, p), "laneA": rmse(y[a], p[a]), "laneB": rmse(y[~a], p[~a])}
    for m in HOLDOUT:
        k = f["month"].values == m
        if k.any():
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
        p, it = outer_predict(cfg, tr[~va], tr[va])
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
