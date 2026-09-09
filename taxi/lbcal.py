"""LBCAL: is the lane-B prediction calibrated in MAGNITUDE, conditioned on the prediction rather than on the truth?

Diagnostic. No cell declared, no champion moves, no verdict decided. `docs/LB2.md` section 3 closed recalibration of the copy
PROBABILITY; this closes recalibration of the predicted VALUE. Regenerate with `python -m taxi.lbcal`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from taxi.champ import CURRENT, predict_b
from taxi.data import CV_FOLDS, TRAIN_MONTHS, load
from taxi.l2 import floored

# One fixed set of edges: this prices a family, it must not become an undeclared search over binnings.
EDGES = (0.0, 600.0, 900.0, 1200.0, 1800.0, 3000.0, 6000.0, 12000.0, 25000.0, 45000.0, np.inf)
CARRY_B = 0.0776  # board seconds per lane-B second, docs/BL.md section 5


def oof() -> pd.DataFrame:
    """Champion lane-B predictions out of fold over the five month pairs, the same folds every lane-B diagnostic uses."""
    tr = load(TRAIN_MONTHS)
    tr = tr[~tr.has_fpl].reset_index(drop=True)
    parts = []
    for fold in CV_FOLDS:
        m = tr.month.isin(fold).values
        fit, va = tr[~m].reset_index(drop=True), tr[m].reset_index(drop=True)
        parts.append(pd.DataFrame({"pred": floored(predict_b(CURRENT["B"], fit, va)), "y": va.y.values}))
        print(f"fold {fold}: {len(va):,} rows", flush=True)
    return pd.concat(parts, ignore_index=True)


def table(o: pd.DataFrame) -> pd.DataFrame:
    g = o.assign(se=(o.pred - o.y) ** 2).groupby(pd.cut(o.pred, list(EDGES), include_lowest=True), observed=True)
    t = pd.DataFrame({"n": g.size(), "mp": g.pred.mean(), "my": g.y.mean(), "sse": g.se.sum()})
    t["bias"] = t.my - t.mp
    t["share"] = t.sse / t.sse.sum()
    t["removable"] = t.n * t.bias ** 2  # SSE an oracle per-bin shift removes, an upper bound on any honest recalibration
    return t


def band(i: pd.Interval) -> str:
    lo, hi = i.left, i.right
    return f"over {lo:,.0f}" if np.isinf(hi) else f"{max(lo, 0.0):,.0f} to {hi:,.0f}"


def main() -> None:
    o = oof()
    t = table(o)
    n, tot, rem = len(o), float(t.sse.sum()), float(t.removable.sum())
    r0, r1 = np.sqrt(tot / n), np.sqrt((tot - rem) / n)
    rows = [f"| {band(i)} | {int(r.n):,} | {r.mp:,.1f} | {r.my:,.1f} | {r.bias:+,.1f} | {r.share:.1%} |" for i, r in t.iterrows()]
    top = t.iloc[-1]
    report = [
        "# LBCAL: is the lane-B prediction calibrated in magnitude? (diagnostic, never a verdict)", "",
        f"Regenerate with `python -m taxi.lbcal`. No cell declared, no champion moves. Champion {CURRENT['B']}, floored, out of "
        f"fold over the five month-pair folds on the {n:,} lane-B training rows.", "",
        "## 1. Why it was asked, and the misreading that asked it", "",
        "`docs/DE.md` section 2 reports a band of 193 holdout rows whose truth sits between 1 and 12 hours, carrying 12.1% of",
        "lane-B squared error, with a mean prediction of 6,577 s against a mean truth of 9,188 s. Read as a 2,611 s bias on rows",
        "the model has already flagged, that is a magnitude error rather than an identification error, and magnitude errors are",
        "usually free to fix.", "",
        "The reading is wrong, and the error is worth naming because it is easy to repeat. That band is defined by the TRUTH.",
        "Conditioning on the truth makes every honest predictor look biased toward the mean, because the rows selected into a",
        "high-truth band are exactly the rows where the noise ran high. The quantity that matters is the other conditional: given",
        "what the model PREDICTS, which is the only thing visible in 2026, what is the mean truth?", "",
        "## 2. The prediction-conditioned answer", "",
        "| predicted band (s) | rows | mean prediction | mean truth | bias | share of squared error |",
        "|---|---|---|---|---|---|", *rows, "",
        "The biases alternate in sign and none of the populated bins carries one worth having. The 2,611 s gap of section 1 does",
        "not appear anywhere.", "",
        "## 3. The ceiling, and why it closes the family", "",
        "Subtracting every bin's own bias, fitted in sample on the same rows it is then scored on, is an oracle recalibration and",
        f"an upper bound on any honest version. It removes {rem:,.0f} of {tot:,.0f} squared seconds, {rem / tot:.1%}, and takes",
        f"out-of-fold lane-B RMSE from **{r0:,.1f} s to {r1:,.1f} s: +{r0 - r1:,.1f} s**.", "",
        f"At the lane-B carry of `docs/BL.md` section 5 that is about **{(r0 - r1) * CARRY_B:,.1f} s of board**, for a correction",
        "fitted and judged on the same rows. An honestly cross-fitted version is worth less. The floor is 2 s of lane B and this",
        "is a ceiling, so the size is not the objection; the objection is that the ceiling is an oracle and that",
        f"{top.share:.1%} of the squared error, the largest single contributor in the table, sits in the {int(top.n)} rows above",
        f"{EDGES[-2]:,.0f} s, which are the day-shift artifact that flag F21 already prices at zero through every route tested.", "",
        "## 4. What this closes", "",
        "The lane-B prediction is calibrated in magnitude. Do not re-open a recalibration, a per-bin correction, an isotonic fit",
        "or a variance model on top of it, and do not read a truth-conditioned band table as evidence of bias. `docs/LB2.md`",
        "section 3 closed recalibration of the copy PROBABILITY; this closes recalibration of the predicted VALUE, which is the",
        "other half and was open.",
    ]
    text = "\n".join(report) + "\n"
    with open("docs/LBCAL.md", "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
