"""X1: XGBoost on the GPU in the M2 recipe, the third and last GBM library. One cell, k = 1; the challenger is the equal-weight bag with SB3."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import xgboost as xgb

from taxi.champ import COLS_D1, lane_a
from taxi.data import CV_FOLDS, TRAIN_MONTHS
from taxi.l2 import floored
from taxi.m1 import CAT
from taxi.m2 import is_tail
from taxi.sb3 import predict as sb3_predict
from taxi.verdict import paired_delta, rmse
from taxi.w1 import verdict_line

CELL = dict(max_depth=8, learning_rate=0.1, reg_lambda=3.0, max_bin=254, random_state=0,
            device="cuda", tree_method="hist", enable_categorical=True)
MAX_ROUNDS, PATIENCE, K = 5000, 100, 1
Models = tuple[xgb.XGBClassifier, xgb.XGBRegressor, xgb.XGBRegressor]


def matrix(df: pd.DataFrame, cols: list[str], levels: dict[str, pd.CategoricalDtype]) -> pd.DataFrame:
    """XGBoost keys on category codes and each frame carries its own levels, so the fit frame's are imposed here; an unseen level goes missing."""
    x = df[cols].copy()
    for c, dt in levels.items():
        x[c] = x[c].astype(dt)
    return x


def levels_of(fit: pd.DataFrame, cols: list[str]) -> dict[str, pd.CategoricalDtype]:
    """Every categorical the cell reads, apt included: the view leaves it an object column and XGBoost takes only numbers, bools and categories."""
    return {c: pd.CategoricalDtype(fit[c].astype("category").cat.categories) for c in CAT if c in cols}


def stages_xgb(tr: pd.DataFrame, va: pd.DataFrame | None, rounds: tuple[int, int, int], cols: list[str]) -> tuple[Models, dict]:
    lv = levels_of(tr, cols)
    t, out = is_tail(tr), []
    labels = [t.astype(int), tr.c.values[t], tr.c.values[~t]]
    fits = [tr, tr[t], tr[~t]]
    vas = [None, None, None]
    if va is not None:
        vt = is_tail(va)
        vas = [(va, vt.astype(int)), (va[vt], va.c.values[vt]), (va[~vt], va.c.values[~vt])]
    for i, kind in enumerate(("clf", "reg", "reg")):
        maker = xgb.XGBClassifier if kind == "clf" else xgb.XGBRegressor
        kw = dict(n_estimators=rounds[i], **CELL)
        if vas[i] is not None:
            kw["early_stopping_rounds"] = PATIENCE
        m = maker(**kw)
        m.fit(matrix(fits[i], cols, lv), labels[i], verbose=False,
              **({"eval_set": [(matrix(vas[i][0], cols, lv), vas[i][1])]} if vas[i] is not None else {}))
        out.append(m)
    return (out[0], out[1], out[2]), lv


def predict_xgb(models: Models, lv: dict, df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    x = matrix(df, cols, lv)
    rng = lambda m: (0, int(m.best_iteration) + 1) if getattr(m, "best_iteration", None) is not None else None  # noqa: E731
    p = models[0].predict_proba(x, iteration_range=rng(models[0]))[:, 1]
    ct, cb = (models[i].predict(x, iteration_range=rng(models[i])) for i in (1, 2))
    return p * ct + (1 - p) * cb + df.z_aobt.values


def trees(m) -> int:
    return int(m.best_iteration) + 1 if getattr(m, "best_iteration", None) is not None else int(m.n_estimators)


def cv(tr: pd.DataFrame, cols: list[str]) -> tuple[tuple[int, int, int], list[str], float]:
    lines = ["| fold | X1 RMSE (s) | X1 trees (clf, tail, body) | fit s |", "|---|---|---|---|"]
    sq, n, counts = 0.0, 0, []
    for fold in CV_FOLDS:
        m, t0 = tr.month.isin(fold).values, time.time()
        models, lv = stages_xgb(tr[~m], tr[m], (MAX_ROUNDS,) * 3, cols)
        e = predict_xgb(models, lv, tr[m], cols) - tr.y.values[m]
        sq, n = sq + float(e @ e), n + len(e)
        counts.append([trees(b) for b in models])
        lines.append(f"| {fold} | {rmse(e):.1f} | {counts[-1]} | {time.time() - t0:.0f} |")
        print(lines[-1], flush=True)
    r, cvr = tuple(int(x) for x in np.mean(counts, axis=0)), float(np.sqrt(sq / n))
    lines.append(f"\nCV RMSE over the five folds: X1 {cvr:.1f} s; final X1 trees {r}.")
    print(lines[-1], flush=True)
    return r, lines, cvr


def main() -> None:
    cols, (tra, tea) = COLS_D1, lane_a()
    y, t0 = tea.y.values, time.time()
    rounds, lines, cvr = cv(tra, cols)
    models, lv = stages_xgb(tra, None, rounds, cols)
    x1 = floored(predict_xgb(models, lv, tea, cols))
    wall = (time.time() - t0) / 60
    sb3 = sb3_predict(tra, tea)
    bag = floored(0.5 * sb3 + 0.5 * x1)
    e0, e1, ex = sb3 - y, bag - y, x1 - y
    v = paired_delta(e0, e1, tea.day.values)
    rho = float(np.corrcoef(e0, ex)[0, 1])
    r0, rx = rmse(e0), rmse(ex)
    # The bar the declaration priced: the best any weight could do on these two, with the weight read off the holdout itself.
    w = (rx ** 2 - rho * r0 * rx) / (r0 ** 2 + rx ** 2 - 2 * rho * r0 * rx)
    orc = float(np.sqrt(w ** 2 * r0 ** 2 + (1 - w) ** 2 * rx ** 2 + 2 * w * (1 - w) * rho * r0 * rx))
    vt, jan, jul = is_tail(tea), tea.month.values == 1, tea.month.values == 7
    row = lambda n, e: f"| {n} | {rmse(e):.2f} | {rmse(e[jan]):.1f} | {rmse(e[jul]):.1f} | {rmse(e[vt]):.1f} | {rmse(e[~vt]):.1f} |"  # noqa: E731
    cell = {k: v for k, v in CELL.items() if k not in ("device", "tree_method", "enable_categorical")}
    out = ["# X1: XGBoost (GPU) in the M2 recipe, Jan + Jul 2025 development holdout", "",
           f"Train months {TRAIN_MONTHS}; regenerate with `python -m taxi.x1`. XGBoost {xgb.__version__}, cell {cell}, up to {MAX_ROUNDS} rounds, "
           f"early stopping {PATIENCE}, on {len(cols)} columns, categoricals native with the fit frame's levels imposed on the holdout. "
           f"The challenger is the equal-weight mean of SB3 and X1, floored, no weight chosen anywhere, k = {K}. "
           f"CV plus the final fit took {wall:.1f} min of wall clock.", "",
           "## Month-pair CV", ""] + lines + ["", f"## Holdout vs champion SB3 ({len(tea):,} rows; tail rows {int(vt.sum()):,})", "",
           "| model | RMSE | Jan | Jul | tail rows | body rows |", "|---|---|---|---|---|---|",
           row("SB3 (champion)", e0), row("X1 alone (diagnostic)", ex), row("SB3 + X1 equal bag (challenger)", e1), "",
           "## The bar the declaration priced", "",
           f"X1's holdout error correlates {rho:.4f} with SB3's, at {rx:.2f} s against SB3's {r0:.2f} s. The best weight on these two, "
           f"read off the holdout itself and so unavailable in practice, is {w:.3f} and reaches {orc:.2f} s, a gain of {r0 - orc:+.2f} s. "
           f"LightGBM's measured point on the same axis was 0.9589 at 241.35 s for +0.07 s.", "",
           verdict_line("SB3 + X1 bag", "SB3", v, k=K)]
    text = "\n".join(out) + "\n"
    with open("docs/X1.md", "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
