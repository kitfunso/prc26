"""L2: bounded lane-B fit. Two declared cells on the no-plan rows, chosen by month-pair CV; the CV winner is judged against floored B4 on Jan+Jul."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor

from taxi.baselines import b1, b4
from taxi.data import CV_FOLDS, HOLDOUT_MONTHS, TRAIN_MONTHS, load
from taxi.verdict import paired_delta, rmse
from taxi.w1 import verdict_line

K = 2
COLS = ["z_sched", "queue"]
Cell = Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]


def floored(p: np.ndarray) -> np.ndarray:
    return np.maximum(p, 0.0)


def cap(tr: pd.DataFrame, te: pd.DataFrame) -> np.ndarray:
    """L2a: B4's per-airport OLS, each airport's prediction capped at that airport's training maximum of y."""
    top = te.apt.map(tr.groupby("apt").y.max()).fillna(tr.y.max()).values
    return np.minimum(b4(tr, te, "B"), top)


def huber(tr: pd.DataFrame, te: pd.DataFrame) -> np.ndarray:
    """L2b: the same two columns fit per airport with a Huber loss; thin airports fall back to B1 exactly as b4 does."""
    out = b1(tr, te, "B")
    for apt, ix in te.groupby("apt").indices.items():
        t = tr[tr.apt == apt]
        if len(t) <= len(COLS) + 1:
            continue
        mu, sd = t[COLS].mean().values, t[COLS].std().values
        sd = np.where(sd > 0, sd, 1.0)
        m = HuberRegressor(epsilon=1.35, alpha=0.0, max_iter=1000).fit((t[COLS].values - mu) / sd, t.y.values)
        out[ix] = m.predict((te[COLS].values[ix] - mu) / sd)
    return out


CELLS: dict[str, Cell] = {"L2a": cap, "L2b": huber}


def cv(trb: pd.DataFrame) -> tuple[str, list[str]]:
    names = ["B4", *CELLS]
    lines = ["| fold | " + " | ".join(f"{k} RMSE" for k in names) + " |", "|---|" + "---|" * len(names)]
    sq, n = {k: 0.0 for k in names}, 0
    for fold in CV_FOLDS:
        m = trb.month.isin(fold).values
        tr, te, row = trb[~m], trb[m].reset_index(drop=True), []
        for k in names:
            e = floored(b4(tr, te, "B") if k == "B4" else CELLS[k](tr, te)) - te.y.values
            sq[k], row = sq[k] + float(e @ e), row + [f"{rmse(e):.1f}"]
        n += len(te)
        lines.append(f"| {fold} | " + " | ".join(row) + " |")
        print(lines[-1], flush=True)
    cvr = {k: float(np.sqrt(sq[k] / n)) for k in names}
    best = min(CELLS, key=cvr.get)
    lines.append("\nCV RMSE over the five folds (every cell floored at 0 s): " + ", ".join(f"{k} {v:.1f} s" for k, v in cvr.items()) + f"; best declared cell {best}.")
    print(lines[-1], flush=True)
    return best, lines


def main() -> None:
    tr, te = load(TRAIN_MONTHS), load(HOLDOUT_MONTHS)
    trb, teb = (d[~d.has_fpl].reset_index(drop=True) for d in (tr, te))
    best, lines = cv(trb)
    p0, p1 = floored(b4(trb, teb, "B")), floored(CELLS[best](trb, teb))
    e0, e1 = p0 - teb.y.values, p1 - teb.y.values
    v = paired_delta(e0, e1, teb.day.values)
    jan, jul = teb.month.values == 1, teb.month.values == 7
    row = lambda name, e: f"| {name} | {rmse(e):.1f} | {rmse(e[jan]):.1f} | {rmse(e[jul]):.1f} | {e.min():.0f} | {e.max():.0f} |"
    report = ["# L2: bounded lane-B fit, Jan+Jul 2025 development holdout", "",
              f"Train months {TRAIN_MONTHS}; regenerate with `python -m taxi.l2`. Cells: L2a B4's per-airport OLS on {COLS} capped at the airport's "
              "training maximum of y; L2b the same columns fit per airport with a Huber loss (scikit-learn HuberRegressor, epsilon 1.35, alpha 0, "
              f"columns standardised). Every cell and the champion carry the floor at 0 s (H22). k = {K}; the CV winner alone meets the holdout.", "",
              "## Month-pair CV (lane-B rows of the fold)", ""] + lines + ["",
              f"## Holdout verdict vs champion B4 with the floor ({len(teb)} rows; {best} differs from it on {int((p1 != p0).sum())} rows)", "",
              "| model | RMSE | Jan | Jul | min error | max error |", "|---|---|---|---|---|---|", row("B4 floored", e0), row(f"{best} floored", e1), "",
              verdict_line(best, "B4", v, k=K)]
    text = "\n".join(report) + "\n"
    with open("docs/L2.md", "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
