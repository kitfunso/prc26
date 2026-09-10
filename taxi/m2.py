"""M2 (protocol section 4): lane-A two-stage tail model, P(|c| > 600 s) classifier with tail and body regressors blended by that probability."""
from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from taxi.data import CV_FOLDS, HOLDOUT_MONTHS, TRAIN_MONTHS, load
from taxi.m1 import BASE, CAT, MAX_ROUNDS, MODEL_FEATURES, PATIENCE, chosen_cells, expected_max, matrix, predict, train
from taxi.verdict import paired_delta, rmse

TAIL = 600.0
K = 1  # one declared configuration (M1's chosen lane-A cell for all three boosters), so the optimism bound is zero
Models = tuple[lgb.Booster, lgb.Booster, lgb.Booster]


def fit(x: pd.DataFrame, label: np.ndarray, cell: dict, va: tuple[pd.DataFrame, np.ndarray] | None = None,
        rounds: int = MAX_ROUNDS, cols: list[str] = MODEL_FEATURES) -> lgb.Booster:
    params = {**BASE, **cell}
    d = lgb.Dataset(matrix(x, cols), label, categorical_feature=[c for c in CAT if c in cols])
    if va is None:
        return lgb.train(params, d, num_boost_round=rounds)
    dv = lgb.Dataset(matrix(va[0], cols), va[1], reference=d)
    return lgb.train(params, d, num_boost_round=rounds, valid_sets=[dv], callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])


def is_tail(df: pd.DataFrame, tail: float = TAIL) -> np.ndarray:
    return (df.c.abs() > tail).values


def stages(tr: pd.DataFrame, cell: dict, va: pd.DataFrame | None = None,
           rounds: tuple[int, int, int] = (MAX_ROUNDS,) * 3, cols: list[str] = MODEL_FEATURES) -> Models:
    t = is_tail(tr)
    parts: list = [None, None, None]
    if va is not None:
        vt = is_tail(va)
        parts = [(va, vt.astype(float)), (va[vt], va.c.values[vt]), (va[~vt], va.c.values[~vt])]
    clf = fit(tr, t.astype(float), {**cell, "objective": "binary"}, parts[0], rounds[0], cols)
    r_tail = fit(tr[t], tr.c.values[t], cell, parts[1], rounds[1], cols)
    r_body = fit(tr[~t], tr.c.values[~t], cell, parts[2], rounds[2], cols)
    return clf, r_tail, r_body


def predict_m2(models: Models, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Prediction and tail probability; E[c | x] = p * tail regressor + (1 - p) * body regressor, then back to y."""
    x = matrix(df, models[0].feature_name())
    p, ct, cb = (b.predict(x, num_iteration=b.best_iteration or None) for b in models)
    return p * ct + (1 - p) * cb + df.z_aobt.values, p


def cv(tr: pd.DataFrame, cell: dict) -> tuple[tuple[int, int, int], list[str]]:
    lines = ["| fold | M1 RMSE (s) | M2 RMSE (s) | M2 rounds (clf, tail, body) | M1 rounds | mean p on tail / body rows | fit s |",
             "|---|---|---|---|---|---|---|"]
    sq1 = sq2 = 0.0
    n, rounds = 0, []
    for fold in CV_FOLDS:
        m, t0 = tr.month.isin(fold).values, time.time()
        va = tr[m]
        models = stages(tr[~m], cell, va=va)
        p2, prob = predict_m2(models, va)
        bst = train(tr[~m], "A", cell, va=va)
        e1, e2 = predict(bst, va, "A") - va.y.values, p2 - va.y.values
        sq1, sq2, n = sq1 + float(e1 @ e1), sq2 + float(e2 @ e2), n + len(e2)
        rounds.append([b.best_iteration for b in models])
        vt = is_tail(va)
        lines.append(f"| {fold} | {rmse(e1):.1f} | {rmse(e2):.1f} | {rounds[-1]} | {bst.best_iteration} | "
                     f"{prob[vt].mean():.3f} / {prob[~vt].mean():.3f} | {time.time() - t0:.0f} |")
        print(lines[-1], flush=True)
    r = tuple(int(x) for x in np.mean(rounds, axis=0))
    lines.append(f"\nCV RMSE over the five folds on the same rows: M1 {np.sqrt(sq1 / n):.1f} s, M2 {np.sqrt(sq2 / n):.1f} s; "
                 f"final M2 rounds (classifier, tail, body) {r}.")
    print(lines[-1], flush=True)
    return r, lines


def main() -> None:
    tr, te = load(TRAIN_MONTHS), load(HOLDOUT_MONTHS)
    tra, tea = (d[d.has_fpl].reset_index(drop=True) for d in (tr, te))
    cell, m1_rounds = chosen_cells()["A"]
    rounds, lines = cv(tra, cell)
    p2, prob = predict_m2(stages(tra, cell, rounds=rounds), tea)
    e2 = p2 - tea.y.values
    e1 = predict(train(tra, "A", cell, rounds=m1_rounds), tea, "A") - tea.y.values
    v = paired_delta(e1, e2, tea.day.values)
    bound = expected_max(K) * (v.hi - v.lo) / 3.92
    clears = v.win and v.delta >= bound + v.floor
    vt, jan, jul = is_tail(tea), tea.month.values == 1, tea.month.values == 7
    row = lambda name, e: f"| {name} | {rmse(e):.1f} | {rmse(e[jan]):.1f} | {rmse(e[jul]):.1f} | {rmse(e[vt]):.1f} | {rmse(e[~vt]):.1f} |"
    report = ["# M2: two-stage tail model, lane A, Jan+Jul 2025 development holdout", "",
              f"Train months {TRAIN_MONTHS}; regenerate with `python -m taxi.m2`. Threshold |c| > {TAIL:.0f} s; k = {K}: "
              f"M1's chosen cell {cell} for the classifier (binary objective) and both regressors; early stopping {PATIENCE} on the validation fold.", "",
              "## Month-pair CV", ""] + lines + ["",
              f"## Holdout verdict vs champion M1 ({len(tea)} rows; tail rows {int(vt.sum())})", "",
              "| model | RMSE | Jan | Jul | tail rows | body rows |", "|---|---|---|---|---|---|", row("M1", e1), row("M2", e2), "",
              f"Classifier mean p: tail rows {prob[vt].mean():.3f}, body rows {prob[~vt].mean():.3f}; p > 0.5 on {int((prob > 0.5).sum())} rows.", "",
              f"Improvement M2 vs M1: {v.delta:+.1f} s [{v.lo:+.1f}, {v.hi:+.1f}]; paired SE {(v.hi - v.lo) / 3.92:.1f} s; "
              f"optimism bound for k = {K}: {bound:.1f} s; floor {v.floor:.0f} s. "
              f"Verdict: {'WIN, clears bound plus floor, M2 promoted' if clears else ('WIN on the interval but inside bound plus floor, not promoted' if v.win else 'tie or lose, M1 stays')}.", ""]
    text = "\n".join(report) + "\n"
    with open("docs/M2.md", "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
