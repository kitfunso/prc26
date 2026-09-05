"""MC1/MC2: CatBoost on the GPU in the M2 recipe, one declared cell each, k = 1. MC1 on W1-A's columns vs W1-A; MC2 on D1's columns vs D1."""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

from taxi.champ import COLS_D1, COLS_D2, champion_holdout, lane_a
from taxi.data import CV_FOLDS, TRAIN_MONTHS
from taxi.m1 import CAT
from taxi.m2 import K, is_tail
from taxi.verdict import paired_delta, rmse
from taxi.w1 import COLS_A, verdict_line

CELL = dict(depth=8, learning_rate=0.1, l2_leaf_reg=3.0, border_count=254, random_seed=0, task_type="GPU", devices="0", verbose=0, allow_writing_files=False)
MAX_ITER, PATIENCE = 5000, 100
# Lane name: its columns and the champion it was declared against.
LANES = {"MC1": (COLS_A, "W1-A"), "MC2": (COLS_D1, "D1"), "D2": (COLS_D2, "MC2")}
Models = tuple[CatBoostClassifier, CatBoostRegressor, CatBoostRegressor]


def pool(df: pd.DataFrame, cols: list[str], label: np.ndarray | None = None) -> Pool:
    x = df[cols].copy()
    cats = [c for c in CAT if c in cols]
    for c in cats:
        x[c] = x[c].astype(str)  # CatBoost takes strings for categorical columns; a missing value becomes its own level
    return Pool(x, label, cat_features=cats)


def fit(kind: str, x: pd.DataFrame, label: np.ndarray, va: tuple[pd.DataFrame, np.ndarray] | None, iters: int, cols: list[str]):
    model = (CatBoostClassifier if kind == "clf" else CatBoostRegressor)(
        **CELL, iterations=iters, loss_function="Logloss" if kind == "clf" else "RMSE")
    if va is None:
        model.fit(pool(x, cols, label))
    else:
        model.fit(pool(x, cols, label), eval_set=pool(va[0], cols, va[1]), early_stopping_rounds=PATIENCE, use_best_model=True)
    return model


def stages_cb(tr: pd.DataFrame, va: pd.DataFrame | None = None, iters: tuple[int, int, int] = (MAX_ITER,) * 3,
              cols: list[str] = COLS_A) -> Models:
    t = is_tail(tr)
    parts: list = [None, None, None]
    if va is not None:
        vt = is_tail(va)
        parts = [(va, vt.astype(int)), (va[vt], va.c.values[vt]), (va[~vt], va.c.values[~vt])]
    clf = fit("clf", tr, t.astype(int), parts[0], iters[0], cols)
    r_tail = fit("reg", tr[t], tr.c.values[t], parts[1], iters[1], cols)
    r_body = fit("reg", tr[~t], tr.c.values[~t], parts[2], iters[2], cols)
    return clf, r_tail, r_body


def predict_cb(models: Models, df: pd.DataFrame, cols: list[str] = COLS_A) -> tuple[np.ndarray, np.ndarray]:
    x = pool(df, cols)
    p = models[0].predict_proba(x)[:, 1]
    ct, cb = models[1].predict(x), models[2].predict(x)
    return p * ct + (1 - p) * cb + df.z_aobt.values, p


def cv_a(tr: pd.DataFrame, cols: list[str], name: str) -> tuple[tuple[int, int, int], list[str]]:
    lines = [f"| fold | {name} RMSE (s) | {name} trees (clf, tail, body) | fit s |", "|---|---|---|---|"]
    sq, n, trees = 0.0, 0, []
    for fold in CV_FOLDS:
        m, t0 = tr.month.isin(fold).values, time.time()
        va = tr[m]
        models = stages_cb(tr[~m], va=va, cols=cols)
        e = predict_cb(models, va, cols)[0] - va.y.values
        sq, n = sq + float(e @ e), n + len(e)
        trees.append([b.tree_count_ for b in models])
        lines.append(f"| {fold} | {rmse(e):.1f} | {trees[-1]} | {time.time() - t0:.0f} |")
        print(lines[-1], flush=True)
    r = tuple(int(x) for x in np.mean(trees, axis=0))
    lines.append(f"\nCV RMSE over the five folds: {name} {np.sqrt(sq / n):.1f} s ({LANES[name][1]} on the same rows: see its report); final {name} trees {r}.")
    print(lines[-1], flush=True)
    return r, lines


def main(name: str) -> None:
    cols, champ = LANES[name]
    tra, tea = lane_a()
    t0 = time.time()
    trees, lines = cv_a(tra, cols, name)
    e1 = predict_cb(stages_cb(tra, iters=trees, cols=cols), tea, cols)[0] - tea.y.values
    wall = (time.time() - t0) / 60
    e0 = champion_holdout(champ, tra, tea) - tea.y.values
    v = paired_delta(e0, e1, tea.day.values)
    vt, jan, jul = is_tail(tea), tea.month.values == 1, tea.month.values == 7
    row = lambda name, e: f"| {name} | {rmse(e):.1f} | {rmse(e[jan]):.1f} | {rmse(e[jul]):.1f} | {rmse(e[vt]):.1f} | {rmse(e[~vt]):.1f} |"
    cell = {k: v for k, v in CELL.items() if k not in ("verbose", "devices", "allow_writing_files")}
    report = [f"# {name}: CatBoost (GPU) in the M2 recipe, Jan+Jul 2025 development holdout", "",
              f"Train months {TRAIN_MONTHS}; regenerate with `python -m taxi.mc1 {name}`. Cell {cell}, up to {MAX_ITER} iterations, early stopping "
              f"{PATIENCE}, on {len(cols)} columns; k = {K}. Champion {champ} refit at its recorded rounds on the same rows. "
              f"CV plus the final fit took {wall:.1f} min of wall clock.", "",
              "## Month-pair CV", ""] + lines + ["", f"## Holdout verdict vs champion {champ} ({len(tea)} rows; tail rows {int(vt.sum())})", "",
              "| model | RMSE | Jan | Jul | tail rows | body rows |", "|---|---|---|---|---|---|", row(champ, e0), row(name, e1), "",
              verdict_line(name, champ, v)]
    text = "\n".join(report) + "\n"
    with open(f"docs/{name}.md", "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main(sys.argv[1] if sys.argv[1:] else "MC1")
