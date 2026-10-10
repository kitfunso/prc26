"""Mixture model: the recorded off-block is one of the known stamps (AOBT, EOBT, schedule) or a physical time.
yhat = sum_k P(k|x) * z_k + P(other|x) * mu(x). `python mix.py cv NAME` / `python mix.py fit NAME IT_CLF IT_REG`."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from data import CACHE, rmse
from features import CAT, DAY, NUM, OCC, WX, build
from train import BASE, FOLDS, HOLDOUT, LGB, RESULTS, fit_predict, report, save_model, stopping_split

Z = ("z_aobt", "z_eobt", "z_sched")
CLF_LGB = dict(objective="multiclass", num_class=4, learning_rate=0.05, num_leaves=127, min_child_samples=50, feature_fraction=0.8,
               bagging_fraction=0.8, bagging_freq=1, lambda_l2=3.0, n_estimators=4000, verbose=-1, random_state=0, n_jobs=12)
CLF_CB = dict(loss_function="MultiClass", task_type="GPU", depth=8, learning_rate=0.1, l2_leaf_reg=3, iterations=4000,
              od_type="Iter", od_wait=200, random_seed=0, verbose=0, thread_count=8, gpu_ram_part=0.5)
CONFIGS: dict[str, dict] = {
    "mix_lgb": dict(clf=dict(family="lgb", params=CLF_LGB), reg=dict(family="lgb", params={**LGB, "n_jobs": 12}, cat=CAT, num=NUM + DAY), cat=CAT, num=NUM + DAY),
    "mix_cb": dict(clf=dict(family="cb", params=CLF_CB), reg=dict(params=BASE, cat=CAT, num=NUM + DAY), cat=CAT, num=NUM + DAY),
    "mix_lgb_occ": dict(clf=dict(family="lgb", params=CLF_LGB), reg=dict(family="lgb", params={**LGB, "n_jobs": 12}, cat=CAT, num=NUM + DAY + OCC), cat=CAT, num=NUM + DAY + OCC),
    "mix_lgb_wx": dict(clf=dict(family="lgb", params=CLF_LGB), reg=dict(family="lgb", params={**LGB, "n_jobs": 12}, cat=CAT, num=NUM + DAY + OCC + WX), cat=CAT, num=NUM + DAY + OCC + WX),
    "mix_lgb_wx_s1": dict(clf=dict(family="lgb", params={**CLF_LGB, "random_state": 1}), reg=dict(family="lgb", params={**LGB, "n_jobs": 12, "random_state": 1}, cat=CAT, num=NUM + DAY + OCC + WX), cat=CAT, num=NUM + DAY + OCC + WX),
    "mix_lgb_wxb": dict(bclf=True, clf=dict(family="lgb", params=CLF_LGB), reg=dict(family="lgb", params={**LGB, "n_jobs": 12}, cat=CAT, num=NUM + DAY + OCC + WX), cat=CAT, num=NUM + DAY + OCC + WX),
    "mix_cb_wx": dict(clf=dict(family="cb", params=CLF_CB), reg=dict(params={**BASE, "thread_count": 8, "gpu_ram_part": 0.5}, cat=CAT, num=NUM + DAY + OCC + WX), cat=CAT, num=NUM + DAY + OCC + WX),
    "mix_cb_wxb": dict(bclf=True, clf=dict(family="cb", params=CLF_CB), reg=dict(params={**BASE, "thread_count": 8, "gpu_ram_part": 0.5}, cat=CAT, num=NUM + DAY + OCC + WX), cat=CAT, num=NUM + DAY + OCC + WX),
}


B_NUM = ["z_sched", "occ", "t_occ", "hour", "wday", "w_tmpf", "w_sknt", "nB_day", "b_share", "h_nB", "n_day", "day_zs_med", "queue", "dep_before", "notyp"]
B_CAT = ["apt", "airline"]
B_CLF = dict(objective="binary", learning_rate=0.03, num_leaves=8, min_child_samples=100, feature_fraction=0.8, bagging_fraction=0.8,
             bagging_freq=1, lambda_l2=10.0, n_estimators=3000, verbose=-1, random_state=0, n_jobs=8)


def fit_bclf(tr: pd.DataFrame, te: pd.DataFrame, va: pd.DataFrame | None, iters: int | None) -> tuple[np.ndarray, int]:
    """Lane B only: P(schedule copy). Small model on the few fields that move it, so LIRF's 1k rows are not drowned by lane A."""
    cols = B_CAT + B_NUM
    b = tr["has_plan"].values == 0
    y = (label(tr[b]) == 2).astype(int)
    n = iters or B_CLF["n_estimators"]
    m = lgb.LGBMClassifier(**{**B_CLF, "n_estimators": n})
    if va is not None:
        vb = va["has_plan"].values == 0
        m.fit(tr[b][cols], y, eval_set=[(va[vb][cols], (label(va[vb]) == 2).astype(int))], categorical_feature=B_CAT,
              callbacks=[lgb.early_stopping(100, verbose=False)])
        return m.predict_proba(te[cols])[:, 1], m.best_iteration_ or n
    m.fit(tr[b][cols], y, categorical_feature=B_CAT)
    return m.predict_proba(te[cols])[:, 1], n


def label(d: pd.DataFrame) -> np.ndarray:
    """0 aobt, 1 eobt, 2 sched, 3 other; a row within 60 s of several stamps takes the earliest-listed one."""
    m = np.full(len(d), 3)
    for k, z in reversed(list(enumerate(Z))):
        m[((d["y"] - d[z]).abs() < 60).values] = k
    return m


def fit_clf(cfg: dict, tr: pd.DataFrame, lab: np.ndarray, te: pd.DataFrame, va: pd.DataFrame | None, lab_va: np.ndarray | None,
            iters: int | None, model_path: Path | None = None) -> tuple[np.ndarray, int]:
    cols, params = cfg["cat"] + cfg["num"], dict(cfg["clf"]["params"])
    if cfg["clf"]["family"] == "lgb":
        n = iters or params.pop("n_estimators")
        m = lgb.LGBMClassifier(**{**params, "n_estimators": n})
        m.fit(tr[cols], lab, eval_set=[(va[cols], lab_va)] if va is not None else None, categorical_feature=cfg["cat"],
              callbacks=[lgb.early_stopping(100, verbose=False)] if va is not None else None)
        save_model(m, model_path, "lgb")
        return m.predict_proba(te[cols]), (m.best_iteration_ or n) if va is not None else n
    if iters:
        params.update(iterations=iters); params.pop("od_type"); params.pop("od_wait")
    m = CatBoostClassifier(**params)
    m.fit(Pool(tr[cols], lab, cat_features=cfg["cat"]), eval_set=Pool(va[cols], lab_va, cat_features=cfg["cat"]) if va is not None else None, use_best_model=va is not None)
    save_model(m, model_path, "cb")
    best = m.get_best_iteration()
    return m.predict_proba(Pool(te[cols], cat_features=cfg["cat"])), best + 1 if best is not None and best >= 0 else params["iterations"]


def predict(cfg: dict, tr: pd.DataFrame, te: pd.DataFrame, va: pd.DataFrame | None = None, iters: tuple[int, ...] | None = None) -> tuple[np.ndarray, tuple[int, ...], np.ndarray]:
    lab, lab_va = label(tr), label(va) if va is not None else None
    P, it_c = fit_clf(cfg, tr, lab, te, va, lab_va, iters[0] if iters else None)
    for k, z in enumerate(Z):
        P[te[z].isna().values, k] = 0
    P /= P.sum(1, keepdims=True)
    mu, it_r = fit_predict(cfg["reg"], tr[lab == 3], te, va[lab_va == 3] if va is not None else None, iters[1] if iters else None)
    its = (it_c, it_r)
    if cfg.get("bclf"):
        ps, it_b = fit_bclf(tr, te, va, iters[2] if iters else None)
        tb = te["has_plan"].values == 0
        P[tb] = np.column_stack([np.zeros(tb.sum()), np.zeros(tb.sum()), ps[tb], 1 - ps[tb]])
        its = (it_c, it_r, it_b)
    yhat = sum(P[:, k] * np.nan_to_num(te[z].values) for k, z in enumerate(Z)) + P[:, 3] * mu
    return yhat, its, P


def cv(name: str) -> None:
    cfg, f = CONFIGS[name], build()
    tr = f[(f["split"] == "train") & ~f["month"].isin(HOLDOUT)]
    ho = f[(f["split"] == "train") & f["month"].isin(HOLDOUT)]
    oof, its, t0 = np.full(len(tr), np.nan), [], time.time()
    for i, months in enumerate(FOLDS):
        va = tr["month"].isin(months).values
        inner, stop = stopping_split(tr[~va])
        _, it, _ = predict(cfg, inner, stop, stop)
        p, _, _ = predict(cfg, tr[~va], tr[va], iters=it)
        oof[va] = p; its.append(it)
        print(f"fold {i} {months}: {rmse(tr['y'].values[va], p):.2f} at {it} iters, {time.time() - t0:.0f}s", flush=True)
    it = tuple(int(np.mean([x[j] for x in its]) * 1.1) for j in range(len(its[0])))
    ph, _, P = predict(cfg, tr, ho, iters=it)
    ph = np.maximum(ph, 0)
    pd.DataFrame({"id": tr["id"].values, "oof": oof}).to_parquet(CACHE / f"{name}_oof.parquet")
    pd.DataFrame({"id": ho["id"].values, "p": ph, "p_aobt": P[:, 0], "p_eobt": P[:, 1], "p_sched": P[:, 2]}).to_parquet(CACHE / f"{name}_holdout.parquet")
    line = f"| {name} | {json.dumps(report(tr['y'].values, oof, tr))} | {json.dumps(report(ho['y'].values, ph, ho))} | {it} | {time.time() - t0:.0f}s |\n"
    RESULTS.open("a", encoding="utf-8").write(line)
    print(line)


def fit(name: str, out: str, iters: tuple[int, ...]) -> None:
    cfg, f = CONFIGS[name], build()
    tr, rk = f[f["split"] == "train"], f[f["split"] == "rank"]
    p, _, _ = predict(cfg, tr, rk, iters=iters)
    pd.DataFrame({"MVT_ID_mvt": rk["id"].values, "TAXITIME_SEC_mvt": np.maximum(p, 0)}).to_parquet(out)
    print(out, len(p), float(np.mean(p)))


if __name__ == "__main__":
    {"cv": lambda: cv(sys.argv[2]), "fit": lambda: fit(sys.argv[2], f"data/{sys.argv[2]}_rank.parquet", tuple(int(x) for x in sys.argv[3:]))}[sys.argv[1]]()
