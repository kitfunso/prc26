"""ST1: a row-dependent blend of MC2 and D1.

TA measured a row-by-row oracle over these two models at 214.3 s where the best CONSTANT weight scores 233.5, worse than MC2 alone at 232.2. That
gap says the constant-weight family is badly specified, not that the band is empty, so here the weight is gated on the row. Three cells, each chosen
out of fold on the cached month-pair OOF predictions, and only the OOF winner meets the holdout.
"""
from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from taxi.champ import champion_holdout, champion_oof, lane_a
from taxi.data import CV_FOLDS
from taxi.e2 import blend, pick
from taxi.m1 import chosen_cells
from taxi.m2 import is_tail
from taxi.verdict import paired_delta, rmse
from taxi.w1 import verdict_line

WEIGHTS = tuple(round(0.1 * i, 1) for i in range(11))
CELLS = ("ST1a", "ST1b", "ST1c")
K = len(CELLS)
META = ["mc2", "d1", "diff", "adiff", "apt", "z_sched", "queue"]


def meta(d: pd.DataFrame, a: np.ndarray, b: np.ndarray, cats: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"mc2": a, "d1": b, "diff": a - b, "adiff": np.abs(a - b),
                         "apt": pd.Categorical(d.apt.values, categories=cats),
                         "z_sched": d.z_sched.values, "queue": d.queue.values})[META]


def fit_cell(name: str, d: pd.DataFrame, a: np.ndarray, b: np.ndarray, y: np.ndarray, cats: list[str]):
    """The cell fitted on these rows only, returned as a predictor plus a one-line description of what it learned."""
    if name == "ST1a":
        raise RuntimeError("ST1a is withdrawn (H53): is_tail reads c = y - z_aobt, so the gate sets each predicted row's weight from that "
                           "row's own label and cannot run on the ranking file. docs/ST1.md keeps the contaminated numbers as history.")
        t = is_tail(d)
        wt, wb = pick(y[t], a[t], b[t], WEIGHTS)[0], pick(y[~t], a[~t], b[~t], WEIGHTS)[0]
        return (lambda d2, a2, b2: np.where(is_tail(d2), blend(a2, b2, wt), blend(a2, b2, wb))), f"weight on MC2: tail rows {wt}, body rows {wb}"
    if name == "ST1c":
        ws = {}
        for ap in cats:
            m = d.apt.values == ap
            if m.sum():
                ws[ap] = pick(y[m], a[m], b[m], WEIGHTS)[0]

        def per_apt(d2: pd.DataFrame, a2: np.ndarray, b2: np.ndarray) -> np.ndarray:
            w = np.array([ws.get(x, 1.0) for x in d2.apt.values])
            return w * a2 + (1 - w) * b2
        return per_apt, "weight on MC2 by airport: " + ", ".join(f"{k} {v}" for k, v in sorted(ws.items()))
    cell, _ = chosen_cells()["A"]
    m = lgb.LGBMRegressor(**cell, n_estimators=400, learning_rate=0.05, verbose=-1)
    m.fit(meta(d, a, b, cats), y, categorical_feature=["apt"])
    return (lambda d2, a2, b2: m.predict(meta(d2, a2, b2, cats))), "LightGBM meta-learner on the two predictions, their difference, airport, schedule gap and queue"


def main() -> None:
    tra, tea = lane_a()
    cats = sorted(tra.apt.unique())
    a_tr, b_tr, y_tr = champion_oof("MC2", tra), champion_oof("D1", tra), tra.y.values
    base = rmse(a_tr - y_tr)
    print(f"MC2 OOF {base:.2f} s, D1 OOF {rmse(b_tr - y_tr):.2f} s over {len(tra):,} rows", flush=True)

    table, learned = [], {}
    for name in CELLS:
        o, t0 = np.full(len(tra), np.nan), time.time()
        for fold in CV_FOLDS:
            m = tra.month.isin(fold).values
            p, _ = fit_cell(name, tra[~m], a_tr[~m], b_tr[~m], y_tr[~m], cats)
            o[m] = p(tra[m], a_tr[m], b_tr[m])
        table.append((name, rmse(o - y_tr), time.time() - t0))
        print(f"| {name} | {table[-1][1]:.2f} | {table[-1][2]:.0f} s |", flush=True)

    best = min(table, key=lambda t: t[1])[0]
    p, learned[best] = fit_cell(best, tra, a_tr, b_tr, y_tr, cats)
    y = tea.y.values
    h2, hd = champion_holdout("MC2", tra, tea), champion_holdout("D1", tra, tea)
    e0, e1 = h2 - y, p(tea, h2, hd) - y
    v = paired_delta(e0, e1, tea.day.values)
    vt, jan, jul = is_tail(tea), tea.month.values == 1, tea.month.values == 7
    row = lambda n, e: f"| {n} | {rmse(e):.1f} | {rmse(e[jan]):.1f} | {rmse(e[jul]):.1f} | {rmse(e[vt]):.1f} | {rmse(e[~vt]):.1f} |"  # noqa: E731

    out = ["# ST1: a row-dependent blend of MC2 and D1, Jan + Jul 2025 development holdout", "",
           "Regenerate with `python -m taxi.st1`. Both base models are cached, so this run touches no GPU. The weight is chosen out of fold over the "
           f"ten training months ({len(tra):,} rows) and only the OOF winner meets the holdout, priced at k = {K}.", "",
           f"MC2 OOF {base:.2f} s, D1 OOF {rmse(b_tr - y_tr):.2f} s on the same rows.", "",
           "| cell | out-of-fold RMSE (s) | fit s |", "|---|---|---|"]
    out += [f"| {n} | {r:.2f} | {s:.0f} |" for n, r, s in table]
    out += ["", f"OOF winner: **{best}**, refit on all ten training months. What it learned: {learned[best]}.", "",
            f"## Holdout verdict vs champion MC2 ({len(tea):,} rows; tail rows {int(vt.sum()):,})", "",
            "| model | RMSE | Jan | Jul | tail rows | body rows |", "|---|---|---|---|---|---|",
            row("MC2", e0), row("D1", hd - y), row(best, e1), "", verdict_line(best, "MC2", v, k=K)]
    text = "\n".join(out) + "\n"
    with open("docs/ST1.md", "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
