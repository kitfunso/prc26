"""Lane A on the residual c = y - z_aobt: tail classifier (|c| > 600), tail and body regressors, z_aobt added back.
Lane B rows come from a donor mixture config. `python lanea.py cv NAME DONOR` / `python lanea.py fit NAME DONOR IT_CLF IT_TAIL IT_BODY`."""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

from data import CACHE, rmse
from features import ARR, CAT, DAY, NUM, OCC, PW, SNOW, WX, WZ, build, with_pw
from mix import CLF_CB, CLF_LGB, fit_clf
from train import BASE, FOLDS, HOLDOUT, LGB, RESULTS, fit_predict, report

TAIL = 600.0
REG = {**BASE, "thread_count": 8, "gpu_ram_part": 0.5}
CLF_LGB_BIN = {**{k: v for k, v in CLF_LGB.items() if k != "num_class"}, "objective": "binary"}
CONFIGS: dict[str, dict] = {
    "ra_lgb": dict(clf=dict(family="lgb", params=CLF_LGB_BIN), reg=dict(family="lgb", params={**LGB, "n_jobs": 12}, cat=CAT, num=NUM + DAY + OCC + WX), cat=CAT, num=NUM + DAY + OCC + WX),
    "ra_lgb_s1": dict(clf=dict(family="lgb", params={**CLF_LGB_BIN, "random_state": 1}), reg=dict(family="lgb", params={**LGB, "n_jobs": 12, "random_state": 1}, cat=CAT, num=NUM + DAY + OCC + WX), cat=CAT, num=NUM + DAY + OCC + WX),
    "ra_cb": dict(clf=dict(family="cb", params={**CLF_CB, "loss_function": "Logloss"}), reg=dict(params=REG, cat=CAT, num=NUM + DAY + OCC + WX), cat=CAT, num=NUM + DAY + OCC + WX),
    "ra_cb_s1": dict(clf=dict(family="cb", params={**CLF_CB, "loss_function": "Logloss", "random_seed": 1}), reg=dict(params={**REG, "random_seed": 1}, cat=CAT, num=NUM + DAY + OCC + WX), cat=CAT, num=NUM + DAY + OCC + WX),
    "ra_cb_s2": dict(clf=dict(family="cb", params={**CLF_CB, "loss_function": "Logloss", "random_seed": 2}), reg=dict(params={**REG, "random_seed": 2}, cat=CAT, num=NUM + DAY + OCC + WX), cat=CAT, num=NUM + DAY + OCC + WX),
    "ra_cb_arr": dict(clf=dict(family="cb", params={**CLF_CB, "loss_function": "Logloss"}), reg=dict(params=REG, cat=CAT, num=NUM + DAY + OCC + WX + ARR), cat=CAT, num=NUM + DAY + OCC + WX + ARR),
    "ra_cb_s1_arr": dict(clf=dict(family="cb", params={**CLF_CB, "loss_function": "Logloss", "random_seed": 1}), reg=dict(params={**REG, "random_seed": 1}, cat=CAT, num=NUM + DAY + OCC + WX + ARR), cat=CAT, num=NUM + DAY + OCC + WX + ARR),
    "ra_cb_s2_arr": dict(clf=dict(family="cb", params={**CLF_CB, "loss_function": "Logloss", "random_seed": 2}), reg=dict(params={**REG, "random_seed": 2}, cat=CAT, num=NUM + DAY + OCC + WX + ARR), cat=CAT, num=NUM + DAY + OCC + WX + ARR),
    "ra_lgb_arr": dict(clf=dict(family="lgb", params=CLF_LGB_BIN), reg=dict(family="lgb", params={**LGB, "n_jobs": 12}, cat=CAT, num=NUM + DAY + OCC + WX + ARR), cat=CAT, num=NUM + DAY + OCC + WX + ARR),
    "ra_cb_wz": dict(clf=dict(family="cb", params={**CLF_CB, "loss_function": "Logloss"}), reg=dict(params=REG, cat=CAT, num=NUM + DAY + OCC + WX + ARR + WZ), cat=CAT, num=NUM + DAY + OCC + WX + ARR + WZ),
    "ra_lgb_wz": dict(clf=dict(family="lgb", params=CLF_LGB_BIN), reg=dict(family="lgb", params={**LGB, "n_jobs": 12}, cat=CAT, num=NUM + DAY + OCC + WX + ARR + WZ), cat=CAT, num=NUM + DAY + OCC + WX + ARR + WZ),
    "ra_cb_pw": dict(clf=dict(family="cb", params={**CLF_CB, "loss_function": "Logloss"}), reg=dict(params=REG, cat=CAT, num=NUM + DAY + OCC + WX + ARR + WZ + PW), cat=CAT, num=NUM + DAY + OCC + WX + ARR + WZ + PW),
    "ra_lgb_pw": dict(clf=dict(family="lgb", params=CLF_LGB_BIN), reg=dict(family="lgb", params={**LGB, "n_jobs": 12}, cat=CAT, num=NUM + DAY + OCC + WX + ARR + WZ + PW), cat=CAT, num=NUM + DAY + OCC + WX + ARR + WZ + PW),
    "ra_lgb_sn": dict(clf=dict(family="lgb", params=CLF_LGB_BIN), reg=dict(family="lgb", params={**LGB, "n_jobs": 12}, cat=CAT, num=NUM + DAY + OCC + WX + ARR + WZ + PW + SNOW), cat=CAT, num=NUM + DAY + OCC + WX + ARR + WZ + PW + SNOW),
    "ra_ridge": dict(clf=dict(family="lgb", params=CLF_LGB_BIN), reg=dict(family="ridge", params=dict(alpha=10.0, solver="lsqr", random_state=0), cat=CAT, num=NUM + DAY + OCC + WX), cat=CAT, num=NUM + DAY + OCC + WX),
    "ra_cb_sn": dict(clf=dict(family="cb", params={**CLF_CB, "loss_function": "Logloss"}), reg=dict(params=REG, cat=CAT, num=NUM + DAY + OCC + WX + ARR + WZ + SNOW), cat=CAT, num=NUM + DAY + OCC + WX + ARR + WZ + SNOW),
}


def _c(d: pd.DataFrame) -> pd.DataFrame:
    return d.assign(y=d["y"] - d["z_aobt"])


def predict(cfg: dict, tr: pd.DataFrame, te: pd.DataFrame, va: pd.DataFrame | None = None, iters: tuple[int, ...] | None = None) -> tuple[np.ndarray, tuple[int, ...]]:
    tr, va = _c(tr), _c(va) if va is not None else None
    t = (tr["y"].abs() > TAIL).values
    tv = (va["y"].abs() > TAIL).values if va is not None else None
    it = iters or (None, None, None)
    P, i0 = fit_clf(cfg, tr, t.astype(int), te, va, tv.astype(int) if va is not None else None, it[0])
    ct, i1 = fit_predict(cfg["reg"], tr[t], te, va[tv] if va is not None else None, it[1])
    cb, i2 = fit_predict(cfg["reg"], tr[~t], te, va[~tv] if va is not None else None, it[2])
    return P[:, 1] * ct + (1 - P[:, 1]) * cb + te["z_aobt"].values, (i0, i1, i2)


def _merge_b(a: pd.DataFrame, donor: pd.DataFrame, rows: pd.DataFrame, col: str) -> pd.DataFrame:
    b = donor[donor["id"].isin(rows["id"][rows["has_plan"] == 0])]
    return pd.concat([a, b[["id", col]]], ignore_index=True)


def cv(name: str, donor: str) -> None:
    cfg, f = CONFIGS[name], build()
    if set(PW) & set(cfg["num"]):
        f = with_pw(f)
    f = f[f["split"] == "train"]
    tr, ho = f[~f["month"].isin(HOLDOUT)], f[f["month"].isin(HOLDOUT)]
    ta, ha = tr[tr["has_plan"] == 1], ho[ho["has_plan"] == 1]
    oof, its, t0 = np.full(len(ta), np.nan), [], time.time()
    for i, months in enumerate(FOLDS):
        va = ta["month"].isin(months).values
        p, it = predict(cfg, ta[~va], ta[va], ta[va])
        oof[va] = p; its.append(it)
        print(f"fold {i} {months}: {rmse(ta['y'].values[va], p):.2f} at {it} iters, {time.time() - t0:.0f}s", flush=True)
    it = tuple(int(np.mean([x[j] for x in its]) * 1.1) for j in range(3))
    ph = np.maximum(predict(cfg, ta, ha, iters=it)[0], 0)
    o = _merge_b(pd.DataFrame({"id": ta["id"].values, "oof": oof}), pd.read_parquet(CACHE / f"{donor}_oof.parquet"), tr, "oof")
    h = _merge_b(pd.DataFrame({"id": ha["id"].values, "p": ph}), pd.read_parquet(CACHE / f"{donor}_holdout.parquet"), ho, "p")
    o.to_parquet(CACHE / f"{name}_oof.parquet"); h.to_parquet(CACHE / f"{name}_holdout.parquet")
    o, h = tr.merge(o, on="id"), ho.merge(h, on="id")
    line = f"| {name} (B from {donor}) | {json.dumps(report(o['y'].values, o['oof'].values, o))} | {json.dumps(report(h['y'].values, h['p'].values, h))} | {it} | {time.time() - t0:.0f}s |\n"
    RESULTS.open("a", encoding="utf-8").write(line)
    print(line)


def fit(name: str, donor: str, iters: tuple[int, ...]) -> None:
    cfg, f = CONFIGS[name], build()
    if set(PW) & set(cfg["num"]):
        f = with_pw(f)
    tr, rk = f[(f["split"] == "train") & (f["has_plan"] == 1)], f[f["split"] == "rank"]
    ra = rk[rk["has_plan"] == 1]
    p = np.maximum(predict(cfg, tr, ra, iters=iters)[0], 0)
    d = pd.read_parquet(f"data/{donor}_rank.parquet")
    b = d[d["MVT_ID_mvt"].isin(rk["id"][rk["has_plan"] == 0])]
    out = pd.concat([pd.DataFrame({"MVT_ID_mvt": ra["id"].values, "TAXITIME_SEC_mvt": p}), b], ignore_index=True)
    out.to_parquet(f"data/{name}_rank.parquet")
    print(f"data/{name}_rank.parquet", len(out), float(out["TAXITIME_SEC_mvt"].mean()))


if __name__ == "__main__":
    {"cv": lambda: cv(sys.argv[2], sys.argv[3]), "fit": lambda: fit(sys.argv[2], sys.argv[3], tuple(int(x) for x in sys.argv[4:]))}[sys.argv[1]]()
