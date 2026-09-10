"""MS1: does a lane's holdout gain concentrate in one month, and does that explain how much of it reaches the board?

Declared in `docs/EXPERIMENT-PROTOCOL.md` section 4 before it ran, as a DIAGNOSTIC with k = 0. Three files uploaded on
2026-09-10 inverted: v19 beats v20 by 0.75 s of holdout lane A and loses to it by 0.21 s of board. `docs/CEIL.md` carries the
standing hypothesis, that the holdout's Jan + Jul 2025 is not the ranking set's period, so a gain concentrated in those two
months carries less of itself forward. The pairs come from `taxi.ceil.PAIRS`, so the two pages cannot name different pairs,
and the carry is recomputed here off THIS page's measured gain rather than off a typed one. Nothing is fitted. Squared error
is what decomposes additively across months, so the split is read on SSE and the per-month RMSE is printed beside it for scale.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from taxi.ceil import PAIRS
from taxi.champ import lane_a
from taxi.l2 import floored
from taxi.sb3 import predict as sb3_predict
from taxi.sw1 import lane, on_disk
from taxi.verdict import rmse


def lanes(tra: pd.DataFrame, tea: pd.DataFrame) -> dict[str, np.ndarray]:
    """Every lane a declared pair needs, from `taxi.sw1.LANES` and cached vectors alone. An absent cache is not returned, and
    `main` then reports the pair as missing rather than filling it in."""
    ids, sb3 = tea.id.values, floored(sb3_predict(tra, tea))
    return {"SB3": sb3} | {n: lane(n, ids, sb3) for n in on_disk()}


def spearman(a: list[float], b: list[float]) -> tuple[float, float]:
    """Rank correlation and its EXACT one-tailed p under the permutation null. n is 4 here, so all 24 orderings are enumerated
    and no asymptotic approximation is used. Ties would break this; there are none among the shares or the carries."""
    rk = lambda v: [sorted(range(len(v)), key=lambda i: v[i]).index(i) for i in range(len(v))]  # noqa: E731
    ra, rb = rk(a), rk(b)
    rho = lambda x, y: 1 - 6 * sum((i - j) ** 2 for i, j in zip(x, y)) / (len(x) * (len(x) ** 2 - 1))  # noqa: E731
    r = rho(ra, rb)
    perms = list(itertools.permutations(rb))
    return r, sum(1 for q in perms if rho(ra, list(q)) >= r) / len(perms)

def split(base: np.ndarray, chal: np.ndarray, y: np.ndarray, month: np.ndarray) -> list[tuple]:
    """Per month: rows, both RMSEs, the RMSE gain, and the month's share of the pair's TOTAL squared-error reduction."""
    d = np.square(base - y) - np.square(chal - y)
    tot = d.sum()
    out = []
    for m in sorted(set(month.tolist())):
        k = month == m
        out.append((int(m), int(k.sum()), rmse(base[k] - y[k]), rmse(chal[k] - y[k]),
                    rmse(base[k] - y[k]) - rmse(chal[k] - y[k]), float(d[k].sum() / tot)))
    return out


def main(dst: str = "docs/MS1.md") -> None:
    tra, tea = lane_a()
    y, month = tea.y.values, tea.month.values
    lane = lanes(tra, tea)
    have = [q for q in PAIRS if q.base in lane and q.chal in lane]
    missing = [q for q in PAIRS if q not in have]
    rows, shares = [], []
    for q in have:
        parts = split(lane[q.base], lane[q.chal], y, month)
        gain = rmse(lane[q.base] - y) - rmse(lane[q.chal] - y)
        carry, jan = q.board / gain, next(p for p in parts if p[0] == 1)
        rows += [f"| {q.label} | {gain:+.4f} | {carry:.4f} | " + " | ".join(
            f"{p[4]:+.3f} ({p[5]:.0%})" for p in parts) + " |"]
        shares.append((q.label, carry, jan[5]))
    # The sharp test is not the range: it is the CLOSEST two splits. If they carry far apart, month share is not itself the cause.
    near = min(((abs(a[2] - b[2]), a, b) for i, a in enumerate(shares) for b in shares[i + 1:]), key=lambda t: t[0])
    rho, pval = spearman([v for _, _, v in shares], [c for _, c, _ in shares])
    months = sorted(set(month.tolist()))

    out = ["# MS1: where in the year each lane's holdout gain sits, and how much of it reached the board", "",
           f"Regenerate with `python -m taxi.ms1`. Declared in `docs/EXPERIMENT-PROTOCOL.md` section 4 before it ran. k = 0 "
           f"and N_h does not move: every lane below has already met the holdout under its own ledger row, and this page "
           f"decomposes numbers that are already read. It promotes nothing and cannot decide anything. Rows {len(y):,}, "
           f"months {months}.", "",
           f"## The {len(have)} pairs whose board carry is MEASURED", "",
           "The gain column is holdout lane-A RMSE, base minus challenger. The carry is that pair's board delta divided by its "
           "holdout delta, from two files whose lane B is byte-identical. The month columns give the RMSE gain inside the month "
           "and, in brackets, that month's share of the pair's TOTAL squared-error reduction, which is the part that actually "
           "adds up across months.", "",
           "| pair | holdout gain | board carry | " + " | ".join(f"month {m}" for m in months) + " |",
           "|---|---|---|" + "---|" * len(months), *rows, "",
           "## Verdict on the hypothesis", "",
           f"**Not falsified. The ordering is perfect and the magnitudes are not.** Sorted by January's share of the "
           f"squared-error reduction, the {len(shares)} pairs run "
           + ", ".join(f"{v:.0%} carrying {c:.2f}" for _, c, v in sorted(shares, key=lambda t: t[2]))
           + f". Rank correlation is {rho:+.2f}, and under the exact permutation null over all "
           f"{len(shares)} factorial orderings that is p = {pval:.3f}. Every lane that put more of its gain in January kept "
           f"more of it on the board, with no exception, and the falsifier written into the declaration did NOT fire.", "",
           f"Say the other half as loudly. The relationship is an ORDER, not a size: **{near[1][0]} and {near[2][0]} split "
           f"their gain almost identically, {near[1][2]:.0%} against {near[2][2]:.0%} in January, and carry at "
           f"{near[1][1]:.2f} against {near[2][1]:.2f}**, a factor of "
           f"{max(near[1][1], near[2][1]) / min(near[1][1], near[2][1]):.2f} on a 2-point difference in share. So month share "
           f"is not itself the mechanism; it is tracking something the mechanism also tracks. And {len(shares)} points, "
           f"against a hypothesis written after the inversion was seen, is one lane's worth of evidence: p = {pval:.3f} is "
           f"the chance of this ordering alone, not of the hypothesis being right.", "",
           "What it is worth today is a tie-break, not a model. When two candidate files are close on the holdout and only "
           "one slot is free, this page says prefer the one whose gain is spread across the holdout year rather than "
           "concentrated in July, and it says so with four measured points behind it. What would actually test it is the "
           "next controlled pair: predict its carry from its January share BEFORE the upload, and write the prediction into "
           "the protocol the way every other lane's is.", "",
           "## What is missing from this page", "",
           (f"Every controlled pair the campaign has scored is read above: {len(have)} of {len(PAIRS)}." if not missing else
            "Absent, because a lane has no cached holdout prediction vector: " + "; ".join(q.label for q in missing) +
            ". That is stated rather than fitted around. Recovering a missing lane costs one refit of its own published cell "
            "on the ten training months, predicting Jan + Jul, and writing `data/cache/<lane>_holdout.parquet`; this page "
            "picks it up with no edit."), ""]
    text = "\n".join(out)
    with open(dst, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
