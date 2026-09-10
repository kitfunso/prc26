"""L1: lane-B campaign. Four declared cells on the no-plan rows, chosen by month-pair CV; the best one is judged against B4 on Jan+Jul."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from taxi.baselines import b4
from taxi.data import COUNTS, CV_FOLDS, HOLDOUT_MONTHS, TRAIN_MONTHS, load
from taxi.m1 import MAX_ROUNDS, MODEL_FEATURES, chosen_cells, matrix
from taxi.m2 import fit
from taxi.verdict import paired_delta, rmse
from taxi.w1 import verdict_line

K = 4
HOUR = 3600.0
COLS_C = ["apt", "hour", "wday", "z_sched", "vfr", *COUNTS]  # COUNTS already carries queue
Cell = Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, int | None], tuple[np.ndarray, int]]


def prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index(drop=True).copy()
    df["vfr"] = (df.frule == "V").astype(int)
    return df


def _boost(x: pd.DataFrame, label: np.ndarray, cell: dict, te: pd.DataFrame, te_label: np.ndarray | None, rounds: int | None, cols: list[str]):
    va = None if rounds else (te, te_label)
    bst = fit(x, label, cell, va, rounds or MAX_ROUNDS, cols)
    it = bst.best_iteration or None
    return bst, it, (it or rounds)


def a_vfr(trb, te, tr_all, cell, rounds):
    return b4(trb, te, "B", ["z_sched", "queue", "vfr"]), 0


def b_log(trb, te, tr_all, cell, rounds):
    lab = np.log1p(np.maximum(trb.y.values, 0.0))
    bst, it, used = _boost(trb, lab, cell, te, np.log1p(np.maximum(te.y.values, 0.0)), rounds, MODEL_FEATURES)
    smear = np.exp(lab - bst.predict(matrix(trb, MODEL_FEATURES), num_iteration=it))
    s = pd.Series(smear).groupby(trb.apt.values).mean()
    scale = te.apt.map(s).fillna(smear.mean()).values
    return np.exp(bst.predict(matrix(te, MODEL_FEATURES), num_iteration=it)) * scale - 1.0, used


def c_hour(trb, te, tr_all, cell, rounds):
    t = (trb.y.values > HOUR).astype(float)
    bst, it, used = _boost(trb, t, {**cell, "objective": "binary"}, te, (te.y.values > HOUR).astype(float), rounds, COLS_C)
    p = bst.predict(matrix(te, COLS_C), num_iteration=it)
    g = trb.y.groupby([trb.apt.values, t]).mean()
    m1 = te.apt.map(g.xs(1.0, level=1)).fillna(trb.y.values[t == 1].mean()).values
    m0 = te.apt.map(g.xs(0.0, level=1)).fillna(trb.y.values[t == 0].mean()).values
    return p * m1 + (1 - p) * m0, used


def d_pool(trb, te, tr_all, cell, rounds):
    bst, it, used = _boost(tr_all, tr_all.y.values, cell, te, te.y.values, rounds, MODEL_FEATURES)
    return bst.predict(matrix(te, MODEL_FEATURES), num_iteration=it), used


def cv(trb: pd.DataFrame, tr_all: pd.DataFrame, cells: dict[str, tuple[Cell, dict]]) -> tuple[str, dict[str, int], list[str]]:
    names = ["B4", *cells]
    lines = ["| fold | " + " | ".join(f"{k} RMSE (rounds)" for k in names) + " |", "|---|" + "---|" * len(names)]
    sq, n, rounds = {k: 0.0 for k in names}, 0, {k: [] for k in names}
    for fold in CV_FOLDS:
        m, ma = trb.month.isin(fold).values, tr_all.month.isin(fold).values
        te, row = trb[m], []
        for k in names:
            p, r = (b4(trb[~m], te, "B"), 0) if k == "B4" else cells[k][0](trb[~m], te, tr_all[~ma], cells[k][1], None)
            e = p - te.y.values
            sq[k], row = sq[k] + float(e @ e), row + [f"{rmse(e):.1f} ({r})"]
            rounds[k].append(r)
        n += len(te)
        lines.append(f"| {fold} | " + " | ".join(row) + " |")
        print(lines[-1], flush=True)
    cvr = {k: float(np.sqrt(sq[k] / n)) for k in names}
    best = min(cells, key=cvr.get)
    final = {k: int(np.mean(rounds[k])) for k in cells}
    lines.append("\nCV RMSE over the five folds: " + ", ".join(f"{k} {v:.1f} s" for k, v in cvr.items())
                 + f"; best declared cell {best}; final rounds {final}.")
    print(lines[-1], flush=True)
    return best, final, lines


def main() -> None:
    tr, te = load(TRAIN_MONTHS), load(HOLDOUT_MONTHS)
    trb, teb, tr_all = prep(tr[~tr.has_fpl]), prep(te[~te.has_fpl]), prep(tr)
    cell_a, cell_b = chosen_cells()["A"][0], chosen_cells()["B"][0]
    cells: dict[str, tuple[Cell, dict]] = {"L1a": (a_vfr, {}), "L1b": (b_log, cell_b), "L1c": (c_hour, cell_b), "L1d": (d_pool, cell_a)}
    best, final, lines = cv(trb, tr_all, cells)
    fn, cell = cells[best]
    e1 = fn(trb, teb, tr_all, cell, max(final[best], 1))[0] - teb.y.values
    e0 = b4(trb, teb, "B") - teb.y.values
    v = paired_delta(e0, e1, teb.day.values, lane="B")
    jan, jul = teb.month.values == 1, teb.month.values == 7
    row = lambda name, e: f"| {name} | {rmse(e):.1f} | {rmse(e[jan]):.1f} | {rmse(e[jul]):.1f} | {e.min():.0f} | {e.max():.0f} |"
    report = ["# L1: lane-B campaign, Jan+Jul 2025 development holdout", "",
              f"Train months {TRAIN_MONTHS}; regenerate with `python -m taxi.l1`. Cells: L1a B4 plus the VFR flag; L1b LightGBM on log1p(y) at "
              f"M1's lane-B cell {cell_b} with per-airport smearing; L1c over-1-h classifier at the same cell on {COLS_C} times per-airport class "
              f"means; L1d pooled LightGBM at M1's lane-A cell {cell_a} on label y over both lanes, scored on lane-B rows. k = {K}; the CV winner "
              f"alone meets the holdout.", "", "## Month-pair CV (lane-B rows of the fold)", ""] + lines + ["",
              f"## Holdout verdict vs champion B4 ({len(teb)} rows)", "", "| model | RMSE | Jan | Jul | min error | max error |",
              "|---|---|---|---|---|---|", row("B4", e0), row(best, e1), "", verdict_line(best, "B4", v, k=K)]
    text = "\n".join(report) + "\n"
    with open("docs/L1.md", "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
