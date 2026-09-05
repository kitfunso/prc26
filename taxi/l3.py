"""L3: lane-B mixture lane. Three declared cells on the no-plan rows, chosen by month-pair CV; the CV winner is judged against floored B4 on Jan+Jul."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from taxi.baselines import b4
from taxi.data import CV_FOLDS, HOLDOUT_MONTHS, TRAIN_MONTHS, load
from taxi.l2 import floored
from taxi.m1 import MAX_ROUNDS, MODEL_FEATURES, chosen_cells, matrix
from taxi.m2 import fit
from taxi.verdict import paired_delta, rmse
from taxi.w1 import verdict_line

K = 3
COPY, DAY = 600.0, 3600.0
COLS_LIN = ["z_sched", "queue", "notyp", "occ", "t_occ0", "nz"]
COLS_GBM = [*MODEL_FEATURES, "notyp", "occ", "t_occ"]
MC2_CACHE = "data/cache/mc2_holdout.parquet"
Cell = Callable[[pd.DataFrame, pd.DataFrame, int | None], tuple[np.ndarray, int]]


def prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index(drop=True).copy()
    df["t_occ0"] = df.t_occ.fillna(0.0)
    df["nz"] = df.notyp * df.z_sched
    df["r"] = df.y - df.z_sched
    return df


def regime(df: pd.DataFrame) -> np.ndarray:
    """0 = normal taxi; 1 = block time copied from the schedule (y within COPY of z_sched); 2 = block an hour or more before the schedule."""
    r = df.y.values - df.z_sched.values
    return np.where(np.abs(r) <= COPY, 1, np.where(r > DAY, 2, 0))


def _design(df: pd.DataFrame, apts: list[str]) -> np.ndarray:
    x = np.column_stack([df.z_sched / 3600, df.notyp, df.occ, df.t_occ0 / 3600, df.nz / 3600, df.queue / 10])
    return np.column_stack([x, *(np.asarray(df.apt == a, float) for a in apts)])


def mixture(tr: pd.DataFrame, te: pd.DataFrame, rounds: int | None) -> tuple[np.ndarray, int]:
    """L3a: P(regime) from a multinomial logistic on the new columns, times each regime's own predictor."""
    reg, apts = regime(tr), sorted(tr.apt.unique())
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(_design(tr, apts), reg)
    p = np.zeros((len(te), 3))
    p[:, clf.classes_] = clf.predict_proba(_design(te, apts))
    a1 = float(tr.r.values[reg == 1].mean()) if (reg == 1).any() else 0.0
    mu2 = float(tr.y.values[reg == 2].mean()) if (reg == 2).any() else float(tr.y.mean())
    return p[:, 0] * b4(tr[reg == 0], te, "B") + p[:, 1] * (te.z_sched.values + a1) + p[:, 2] * mu2, 0


def linear(tr: pd.DataFrame, te: pd.DataFrame, rounds: int | None) -> tuple[np.ndarray, int]:
    """L3b: B4's per-airport least squares with the no-type flag, stand occupancy and the flag times z_sched as extra columns."""
    return b4(tr, te, "B", COLS_LIN), 0


def gbm(tr: pd.DataFrame, te: pd.DataFrame, rounds: int | None) -> tuple[np.ndarray, int]:
    """L3c: LightGBM at M1's lane-B cell on M1's columns plus the new ones, label y minus z_sched; early stopping on the fold when rounds is None."""
    cell = chosen_cells()["B"][0]
    bst = fit(tr, tr.r.values, cell, None if rounds else (te, te.r.values), rounds or MAX_ROUNDS, COLS_GBM)
    it = bst.best_iteration or None
    return te.z_sched.values + bst.predict(matrix(te, COLS_GBM), num_iteration=it), it or rounds


CELLS: dict[str, Cell] = {"L3a": mixture, "L3b": linear, "L3c": gbm}


def cv(trb: pd.DataFrame) -> tuple[str, dict[str, int], list[str]]:
    names = ["B4", *CELLS]
    lines = ["| fold | " + " | ".join(f"{k} RMSE" for k in names) + " |", "|---|" + "---|" * len(names)]
    sq, n, rounds = {k: 0.0 for k in names}, 0, {k: [] for k in CELLS}
    for fold in CV_FOLDS:
        m = trb.month.isin(fold).values
        tr, te, row = prep(trb[~m]), prep(trb[m]), []
        for k in names:
            if k == "B4":
                p = b4(tr, te, "B")
            else:
                p, r = CELLS[k](tr, te, None)
                rounds[k].append(r)
            e = floored(p) - te.y.values
            sq[k], row = sq[k] + float(e @ e), row + [f"{rmse(e):.1f}"]
        n += len(te)
        lines.append(f"| {fold} | " + " | ".join(row) + " |")
        print(lines[-1], flush=True)
    cvr = {k: float(np.sqrt(sq[k] / n)) for k in names}
    final = {k: int(np.mean(rounds[k])) for k in CELLS}
    best = min(CELLS, key=cvr.get)
    lines.append("\nCV RMSE over the five folds (every cell floored at 0 s): " + ", ".join(f"{k} {v:.1f} s" for k, v in cvr.items())
                 + f"; best declared cell {best}; final rounds {final}.")
    print(lines[-1], flush=True)
    return best, final, lines


def by_airport(te: pd.DataFrame, errs: dict[str, np.ndarray]) -> list[str]:
    lines = ["| airport | rows | " + " | ".join(f"{k} RMSE" for k in errs) + " | share of B4 squared error |", "|---|---|" + "---|" * (len(errs) + 1)]
    e0 = next(iter(errs.values()))
    for apt, ix in te.groupby("apt").indices.items():
        lines.append(f"| {apt} | {len(ix)} | " + " | ".join(f"{rmse(e[ix]):.0f}" for e in errs.values()) + f" | {float(e0[ix] @ e0[ix]) / float(e0 @ e0):.1%} |")
    return lines


def all_rows(te: pd.DataFrame, eb: np.ndarray) -> str:
    """All-rows holdout RMSE with the cached MC2 lane-A errors: reported, never decides."""
    tea = te[te.has_fpl].merge(pd.read_parquet(MC2_CACHE)[["id", "pred"]], on="id", how="left")
    assert tea.pred.notna().all(), "MC2 cache does not cover the lane-A holdout"
    ea = tea.pred.values - tea.y.values
    return f"{rmse(np.concatenate([ea, eb])):.1f} s (lane A MC2 cached {rmse(ea):.1f} s on {len(ea)} rows, lane B {rmse(eb):.1f} s on {len(eb)} rows)"


def main() -> None:
    tr, te = load(TRAIN_MONTHS), load(HOLDOUT_MONTHS)
    trb, teb = (d[~d.has_fpl].reset_index(drop=True) for d in (tr, te))
    reg = regime(trb)
    best, final, lines = cv(trb)
    trb, teb = prep(trb), prep(teb)
    p0, p1 = floored(b4(trb, teb, "B")), floored(CELLS[best](trb, teb, final[best] or None)[0])
    e0, e1 = p0 - teb.y.values, p1 - teb.y.values
    v = paired_delta(e0, e1, teb.day.values)
    jan, jul = teb.month.values == 1, teb.month.values == 7
    row = lambda name, e: f"| {name} | {rmse(e):.1f} | {rmse(e[jan]):.1f} | {rmse(e[jul]):.1f} | {e.min():.0f} | {e.max():.0f} |"
    counts = ", ".join(f"{name} {int((reg == i).sum())} ({(reg == i).mean():.1%}, mean y {trb.y.values[reg == i].mean():.0f} s)" for i, name in enumerate(("normal", "copy", "day")))
    report = ["# L3: lane-B mixture lane, Jan+Jul 2025 development holdout", "",
              f"Train months {TRAIN_MONTHS}; regenerate with `python -m taxi.l3`. Cells: L3a mixture (multinomial logistic P(regime) on z_sched, the no-type flag, "
              "stand occupancy, flag x z_sched, queue and airport dummies; regimes normal / copy / day; prediction = P(copy) x (z_sched + mean copy offset) + "
              "P(day) x mean y of day rows + P(normal) x B4's per-airport OLS fit on the normal rows); L3b B4's per-airport OLS on "
              f"{COLS_LIN}; L3c LightGBM at M1's lane-B cell on M1's columns plus notyp, occ, t_occ with label y minus z_sched. Every cell and the champion "
              f"carry the floor at 0 s (H22). k = {K}; the CV winner alone meets the holdout.", "",
              f"Training regimes (copy = |y - z_sched| <= {COPY:.0f} s, day = y - z_sched > {DAY:.0f} s): {counts}.", "",
              "## Month-pair CV (lane-B rows of the fold)", ""] + lines + ["",
              f"## Holdout verdict vs champion B4 with the floor ({len(teb)} rows; {best} differs from it on {int((p1 != p0).sum())} rows)", "",
              "| model | RMSE | Jan | Jul | min error | max error |", "|---|---|---|---|---|---|", row("B4 floored", e0), row(f"{best} floored", e1), "",
              verdict_line(best, "B4", v, k=K), "", "## By airport (reported)", ""] + by_airport(teb, {"B4": e0, best: e1}) + ["",
              f"All rows with B4 (reported, never decides): {all_rows(te, e0)}; with {best}: {all_rows(te, e1)}."]
    text = "\n".join(report) + "\n"
    with open("docs/L3.md", "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
