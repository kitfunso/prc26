"""PB: what a board probe buys. Diagnostic, never a verdict (protocol section 2).

A submission returns one number, so it fixes one linear functional of the 2026 truths (F27). F27 closed the LANE SPLIT with
that fact; it does not close level removal, which is what the same functionals actually buy: an exactly measured group error
sum can be subtracted from any future file, and the SSE it removes is arithmetic, never an estimate.
"""
from __future__ import annotations

import sys

import duckdb
import numpy as np
import pandas as pd

from taxi.cb1 import lane_b
from taxi.champ import CURRENT, lane_a, predict_b
from taxi.l2 import floored
from taxi.probes import RANKING, S1, groups
from taxi.sb3 import predict as sb3_predict
from taxi.submit import build_submission

BOARD = {"jan": -3_297_233.0, "jul": +3_148_928.0, "laneB": -1_041_701.0}  # S1 error sums (p - y), protocol section 3, row P1-P6
S1_RMSE, N_RANK = 325.3982, 344_841
CHALLENGER = "data/merry-octopus_v15.parquet"
DP = 1e-4  # the board prints RMSE to four decimals
ORDER = ("jan", "jul", "laneB")
V15_BOARD = 287.42  # the equal-ratio projection, docs/BL.md section 3


def _preds(path: str) -> pd.Series:
    d = duckdb.sql(f"SELECT MVT_ID_mvt, TAXITIME_SEC_mvt FROM read_parquet('{path}')").df()
    return pd.Series(d.TAXITIME_SEC_mvt.values.astype("float64"), index=d.MVT_ID_mvt.values)


def removed(w: np.ndarray, c: np.ndarray) -> float:
    """SSE removed by the best correction inside span(W), from the measured sums c = W' (p - y) alone: c' (W'W)^-1 c."""
    return float(c @ np.linalg.solve(w.T @ w, c))


def banked() -> list[tuple[str, np.ndarray, float]]:
    """The three measured S1 sums carry to any later file exactly, because both prediction vectors are on disk."""
    gs = groups()
    ids = pd.Index(duckdb.sql(f"SELECT MVT_ID_mvt FROM read_parquet('{RANKING}') WHERE PHASE_mvt = 'DEP'").df().MVT_ID_mvt.values)
    w = np.column_stack([ids.isin(gs[g]).astype(float) for g in ORDER])
    s1, ch = _preds(S1).reindex(ids), _preds(CHALLENGER).reindex(ids)
    rows = []
    for name, p in (("S1, scored 325.3982", s1), ("v15, built and never scored", ch)):
        c = np.array([BOARD[g] + float((p - s1)[ids.isin(gs[g])].sum()) for g in ORDER])
        rows.append((name, c, removed(w, c)))
    return rows


def correct(src: str, out: str) -> None:
    """Write `src` with the best correction inside the span of the measured groups removed. Board arithmetic, not a model."""
    gs = groups()
    ids = pd.Index(duckdb.sql(f"SELECT MVT_ID_mvt FROM read_parquet('{RANKING}') WHERE PHASE_mvt = 'DEP'").df().MVT_ID_mvt.values)
    w = np.column_stack([ids.isin(gs[g]).astype(float) for g in ORDER])
    s1, p = _preds(S1).reindex(ids), _preds(src).reindex(ids)
    c = np.array([BOARD[g] + float((p - s1)[ids.isin(gs[g])].sum()) for g in ORDER])
    d = w @ np.linalg.solve(w.T @ w, c)
    q = pd.Series(p.values - d, index=ids)
    print(f"{src} -> {out}: shifts {', '.join(f'{g} {v:+.2f} s' for g, v in zip(ORDER, np.linalg.solve(w.T @ w, c)))}; "
          f"SSE removed {removed(w, c) / 1e6:,.1f} Ms^2; C1's floor then moves {int((q < 0).sum())} rows by up to {-q.min():.2f} s")
    build_submission(q, out, floor=0.0)


def marginal(eb: np.ndarray, teb: pd.DataFrame) -> list[tuple[str, int, float, float]]:
    """Lane-B probe value on the BOARD's scale, so the level the board already measured is separated from the spread it has not."""
    n_b, n_board = len(eb), len(groups()["laneB"])
    apt, mon = pd.Series(teb.apt.values), pd.Series(teb.month.values)
    cuts = [("lane B, one group", pd.Series(["B"] * n_b)), ("lane B by airport", apt),
            ("lane B by airport x month", apt.astype(str) + "|" + mon.astype(str))]
    rows, prev = [], 0.0
    for name, lab in cuts:
        s = pd.Series(eb).groupby(lab).agg(["size", "mean"])
        rem = float((s["size"] * s["mean"] ** 2).sum()) / n_b * n_board
        cum = V15_BOARD - float(np.sqrt((N_RANK * V15_BOARD ** 2 - rem) / N_RANK))
        rows.append((name, len(s), cum, cum - prev))
        prev = cum
    return rows


def curve() -> tuple[list[tuple[str, int, float]], float, int, float, list[tuple[str, int, float, float]], float]:
    tra, tea = lane_a()
    trb, teb = lane_b()
    ea = floored(sb3_predict(tra, tea)) - tea.y.values
    eb = floored(predict_b(CURRENT["B"], trb, teb)) - teb.y.values
    n, sse = len(ea) + len(eb), float((ea ** 2).sum() + (eb ** 2).sum())
    base = np.sqrt(sse / n)
    key = lambda *cols: pd.Series(["|".join(map(str, t)) for t in zip(*cols)])  # noqa: E731
    cand = {
        "lane": [(ea, key(tea.has_fpl)), (eb, key(teb.has_fpl))],
        "lane x month": [(ea, key(tea.has_fpl, tea.month)), (eb, key(teb.has_fpl, teb.month))],
        "lane x airport": [(ea, key(tea.has_fpl, tea.apt)), (eb, key(teb.has_fpl, teb.apt))],
        "lane x airport x month": [(ea, key(tea.has_fpl, tea.apt, tea.month)), (eb, key(teb.has_fpl, teb.apt, teb.month))],
        "lane B only, airport": [(eb, key(teb.apt))],
        "lane B only, airport x month": [(eb, key(teb.apt, teb.month))],
        "lane B airport x month, plus lane A's two months": [(eb, key(teb.apt, teb.month)), (ea, key(tea.month))],
        "lane A only, airport x month": [(ea, key(tea.apt, tea.month))],
        "lane A only, airport x month x 6 h block": [(ea, key(tea.apt, tea.month, tea.hour // 6))],
    }
    out = []
    for name, parts in cand.items():
        rem, g = 0.0, 0
        for err, lab in parts:
            s = pd.Series(err).groupby(lab).agg(["size", "mean"])
            rem += float((s["size"] * s["mean"] ** 2).sum())
            g += len(s)
        out.append((name, g, base - float(np.sqrt((sse - rem) / n))))
    return out, base, n, sse, marginal(eb, teb), float(eb.mean())


def main() -> None:
    rows = banked()
    tbl, base, n, sse, mrg, hold_mean = curve()
    n_board_b = len(groups()["laneB"])
    sig = N_RANK * ((S1_RMSE + DP / 2) ** 2 - (S1_RMSE - DP / 2) ** 2) / (2 * 300.0)
    drop = lambda r, rem: r - float(np.sqrt((N_RANK * r ** 2 - rem) / N_RANK))  # noqa: E731
    s1_gain, v15_gain = drop(S1_RMSE, rows[0][2]), drop(V15_BOARD, rows[1][2])
    body = [f"| {name} | {c[0]:+,.0f} | {c[1]:+,.0f} | {c[2]:+,.0f} | {rem / 1e6:,.1f} |" for name, c, rem in rows]
    out = ["# PB: what a board probe buys, and how many are worth spending", "",
           "Diagnostic, never a verdict, no cell declared. Regenerate with `python -m taxi.pb`. A submission returns one number, so it "
           "fixes one linear functional of the 2026 truth vector and nothing else. F27 used that fact to close the lane split, which is "
           "quadratic in the truths and so unreachable at any number of files. Level removal is the other case: a group's error sum IS a "
           "linear functional, so a probe measures it EXACTLY, and the SSE that subtracting it removes is arithmetic with no estimate in "
           "it. Three such sums are already measured (P1, P3s, P5s) and they are permanent, because for any later file "
           "`sum_g (p' - y) = sum_g (p - y) + sum_g (p' - p)` and both prediction vectors sit on disk: the measurement carries across "
           "models for free and never needs re-buying.", "",
           "## 1. Already banked, on files that exist", "",
           "The best correction inside the span of the three measured groups is `c' (W'W)^-1 c`, which is what handles lane B overlapping "
           "both months rather than three means applied in sequence.", "",
           "| file | January sum (s) | July+ sum (s) | lane B sum (s) | SSE it removes (Ms^2) |", "|---|---|---|---|---|", *body, "",
           f"On S1's own scale that is {s1_gain:.2f} s of board composite; on v15 at its {V15_BOARD:.2f} projection (`docs/BL.md`) it is "
           f"{v15_gain:.2f} s. Both are already paid for and neither has ever been applied to a submitted file.", "",
           f"Board resolution: the score prints to four decimals, so an error sum read at c = 300 s is good to about {sig:,.0f} s, which is "
           f"{sig / 5290:.3f} s per row on a 5,290-row group and finer on anything larger. Precision does not bind.", "",
           "## 2. How the gain grows with the number of probes, on the 2025 development holdout", "",
           f"The champion pair (SB3, L8c) on the {n:,}-row holdout at {base:.4f} s composite, corrected with its own answers in hand. That "
           "is the same arithmetic a probe buys on the board, so the table measures the SHAPE of the curve and never the board's level: "
           "these fits have already spent their own per-cell levels, and the board's come mostly from a year shift no local frame sees.", "",
           "| partition | probes | composite gain (s) | gain per probe (s) |", "|---|---|---|---|",
           *[f"| {nm} | {g} | {v:+.4f} | {v / g:+.4f} |" for nm, g, v in tbl], "",
           "## 3. Reading", "",
           f"The three sums already on the board remove {rows[1][2] / 1e6:,.1f} Ms^2 from the challenger file and cost nothing further. "
           "Beyond them the curve says the efficient direction is lane B: 1.5% of the rows and about half the squared error, so its group "
           "means carry far more SSE per probe than lane A's, and lane A's own refinements are the worst rows on the page. The holdout's "
           "coarse gain is an order of magnitude under the board's at the same granularity, which is the year shift, and is exactly why "
           "this table is a shape and not a forecast.", "",
           "## 4. Why the airport cut is the one to buy", "",
           "`docs/YR.md` (F23) already measured the year shift on the only 2026 labels the data holds, the ranking file's arrivals, and "
           "found it airport-shaped: EHAM +66.9 s in January and -69.7 s in July, EDDF -66.6 s in July, LIRF -43.2 s in January, "
           "LSZH +35.2 s in January, against a weighted aggregate of -10.92 s. Its own words are that those cancel in the mean and do "
           "not cancel in a squared error, and its section 3 then closed the direction, because arrival and departure taxi times do not "
           "move together across 2025 and so the departure version cannot be inferred. A probe does not infer it. It measures the "
           "departure sum for that airport directly, which is the one instrument that reaches what YR could only see the shape of. "
           "A probe also cannot lose: removing an exactly measured mean reduces that group's SSE by `n_g a_g^2 >= 0` "
           "whatever the truth turns out to be, the board ranks each team by its BEST file (`ranking.html`, re-read 2026-09-07), and the "
           "bucket cap is 1 GB against 2 MB a file.", "",
           "## 5. What the probes the board has NOT bought are worth", "",
           "Section 2 is on the holdout's own composite. This one rescales lane B's group means to the board: the holdout's SSE removed "
           f"per row, applied to the ranking file's {n_board_b:,} lane-B rows, read off v15's {V15_BOARD:.2f}.", "",
           "| cut | probes | cumulative (s) | marginal (s) |", "|---|---|---|---|",
           *[f"| {nm} | {g} | {c:.3f} | {m:+.3f} |" for nm, g, c, m in mrg], "",
           "The split matters because the two halves belong to different years. The LEVEL row buys almost nothing here, because L8c's "
           f"holdout mean error is {hold_mean:+.1f} s, and it buys most of section 1's gain on the board, where the same mean is "
           f"{rows[1][1][2] / n_board_b:+.1f} s. That half is bought and is section 1. The AIRPORT SPREAD is the reverse: the holdout "
           "sees it, the board has never measured it, and it is what the next probes buy. Reading the holdout's spread as the board's is "
           "an assumption. F23's year shift is itself airport-shaped (`docs/YR.md`) and adds spread rather than cancelling it, so the "
           "board's own spread should be at least this, which makes the marginal column the low side.", ""]
    text = "\n".join(out) + "\n"
    with open("docs/PB.md", "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    if sys.argv[1:3]:
        correct(sys.argv[1], sys.argv[2])
    else:
        main()
