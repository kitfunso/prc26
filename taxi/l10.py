"""L10: lane B on the residual label. Every lane-B model so far fits raw y, but z_sched is the lane's planned taxi and the whole L3-L8 chain
hand-builds a branch that predicts it exactly, so y - z_sched is the label that makes a copy the zero prediction. M1's lane-B cell and LP1's 24
columns are both inherited, so the only declared change is the label."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from taxi.cb1 import lane_b
from taxi.champ import CURRENT, predict_b
from taxi.data import CV_FOLDS
from taxi.l2 import floored
from taxi.lp1 import COLS
from taxi.m1 import BASE, MAX_ROUNDS, PATIENCE, chosen_cells
from taxi.verdict import paired_delta, rmse
from taxi.w1 import verdict_line

CATS = ["apt", "runway", "stand", "actype", "frule", "airline"]  # airline is L8's derived column, absent from m1.CAT


def levels(fit: pd.DataFrame) -> dict[str, list[str]]:
    return {c: sorted(fit[c].astype(str).unique()) for c in CATS}


def mat(d: pd.DataFrame, lv: dict[str, list[str]]) -> pd.DataFrame:
    """The fit frame's levels on both sides, so a level means the same thing and an unseen one is missing rather than a new code."""
    x = d[COLS].copy()
    for c in CATS:
        x[c] = pd.Categorical(x[c].astype(str), categories=lv[c])
    return x


def run(fit: pd.DataFrame, va: pd.DataFrame, cell: dict, rounds: int | None = None) -> tuple[np.ndarray, int]:
    lv, lab = levels(fit), fit.y.values - fit.z_sched.values
    d = lgb.Dataset(mat(fit, lv), lab, categorical_feature=CATS)
    p = {**BASE, **cell}
    if rounds is None:
        dv = lgb.Dataset(mat(va, lv), va.y.values - va.z_sched.values, reference=d)
        bst = lgb.train(p, d, num_boost_round=MAX_ROUNDS, valid_sets=[dv], callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])
    else:
        bst = lgb.train(p, d, num_boost_round=rounds)
    it = bst.best_iteration or None
    return bst.predict(mat(va, lv), num_iteration=it) + va.z_sched.values, int(it or rounds)


def main() -> None:
    trb, teb = lane_b()
    cell, _ = chosen_cells()["B"]
    print(f"lane B: {len(trb):,} training rows, {len(teb):,} holdout rows, {len(COLS)} columns, cell {cell}", flush=True)

    lines = [f"| fold | rows | {CURRENT['B']} RMSE (s) | L10 RMSE (s) | rounds |", "|---|---|---|---|---|"]
    its = []
    for fold in CV_FOLDS:
        m = trb.month.isin(fold).values
        f, va = trb[~m].reset_index(drop=True), trb[m].reset_index(drop=True)
        p, it = run(f, va, cell)
        its.append(it)
        y = va.y.values
        base = floored(predict_b(CURRENT["B"], f, va))
        lines.append(f"| {fold} | {len(va):,} | {rmse(base - y):.1f} | {rmse(floored(p) - y):.1f} | {it} |")
        print(lines[-1], flush=True)

    rounds = int(np.median(its))
    p, _ = run(trb, teb, cell, rounds)
    l10, base, y = floored(p), floored(predict_b(CURRENT["B"], trb, teb)), teb.y.values
    vd = paired_delta(base - y, l10 - y, teb.day.values)
    jan, jul = teb.month.values == 1, teb.month.values == 7
    row = lambda n, e: f"| {n} | {rmse(e):.1f} | {rmse(e[jan]):.1f} | {rmse(e[jul]):.1f} |"  # noqa: E731

    out = ["# L10: lane B on the residual label, Jan + Jul 2025 development holdout", "",
           f"Regenerate with `python -m taxi.l10`. LightGBM at M1's lane-B cell `{cell}` on LP1's {len(COLS)} columns, label y minus z_sched, "
           f"prediction floored at 0 s. Rounds are the median of the folds below, {rounds}.", "", "## Month-pair folds", ""] + lines
    out += ["", f"## Holdout verdict vs champion {CURRENT['B']} ({len(teb):,} rows)", "",
            "| model | RMSE | Jan | Jul |", "|---|---|---|---|", row(CURRENT["B"], base - y), row("L10", l10 - y), "",
            verdict_line("L10", CURRENT["B"], vd, k=1)]
    text = "\n".join(out) + "\n"
    with open("docs/L10.md", "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
