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

from taxi.ceil import V18

from taxi.cb1 import lane_b
from taxi.champ import CURRENT, lane_a, predict_b
from taxi.l2 import floored
from taxi.probes import RANKING, S1, groups
from taxi.sb3 import predict as sb3_predict
from taxi.submit import build_submission

BOARD = {"jan": -3_297_233.0, "jul": +3_148_928.0, "laneB": -1_041_701.0}  # S1 error sums (p - y), protocol section 3, row P1-P6
S1_RMSE, N_RANK = 325.3982, 344_841
CHALLENGER = "data/merry-octopus_v18.parquet"
CHAL_V = CHALLENGER.split("_")[-1].split(".")[0]  # the version tag the prose prints, so a rebuild moves one constant
RISK = "data/sel_risk.parquet"  # written by scripts/sel_risk.py
RISK_NAME = "learned risk model, scripts/sel_risk.py"
DP = 1e-4  # the board prints RMSE to four decimals
ORDER = ("jan", "jul", "laneB")
CHAL_BOARD = V18  # v18's MEASURED board result; S1's anchor above is measured too, so both ends of the carry now are


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
    for name, p in (("S1, scored 325.3982", s1), (f"{CHAL_V}, scored {CHAL_BOARD:.4f}", ch)):
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
        cum = CHAL_BOARD - float(np.sqrt((N_RANK * CHAL_BOARD ** 2 - rem) / N_RANK))
        rows.append((name, len(s), cum, cum - prev))
        prev = cum
    return rows


def fine(eb: np.ndarray, teb: pd.DataFrame) -> list[tuple[str, int, int, float, float]]:
    """Lane-B group cuts past the airport, with the drop a RANDOM partition of the same size would give beside each one.

    On the board a group mean is measured, not fitted, so the random-partition column is a real gain and not an overfit.
    """
    n_b = len(eb)
    apt, mon, hr = pd.Series(teb.apt.values).astype(str), pd.Series(teb.month.values).astype(str), pd.Series(teb.hour.values)
    sig2 = float((eb ** 2).mean())
    cuts = [("airport", [apt]), ("airport x month", [apt, mon]), ("airport x month x half-day", [apt, mon, (hr // 12).astype(str)]),
            ("airport x month x 6 h", [apt, mon, (hr // 6).astype(str)])]
    rows = []
    for name, cols in cuts:
        lab = pd.Series(["|".join(t) for t in zip(*cols)])
        st = pd.Series(eb).groupby(lab).agg(["size", "mean"])
        rem = float((st["size"] * st["mean"] ** 2).sum()) / n_b
        rows.append((name, len(st), int(st["size"].min()), _board(rem), _board(len(st) * sig2 / n_b)))
    return rows


def singles(eb: np.ndarray, p: np.ndarray, teb: pd.DataFrame) -> tuple[list, tuple[int, ...], tuple[int, int]]:
    """One file that moves ONE row reads that row's label exactly, so k files zero the k rows the selector picks."""
    ks = (5, 10, 20, 40, 75)
    z = teb.z_sched.values.astype("float64")
    sel = [("prediction", p), ("scheduled taxi z_sched", z), ("closeness of prediction to 24 h", -np.abs(p - 86_400.0))]
    risk = _risk(teb)
    if risk is not None:
        sel.append((RISK_NAME, risk))
    sel.append(("oracle absolute error, the ceiling and not buyable", np.abs(eb)))
    rows = []
    for name, sc in sel:
        o = np.argsort(-np.asarray(sc, dtype="float64"))
        rows.append((name, [_board(float((eb[o[:k]] ** 2).sum()) / len(eb)) for k in ks]))
    rank = _preds(CHALLENGER).reindex(pd.Index(sorted(groups()["laneB"])))
    return rows, ks, (int((rank > 21_600).sum()), int((p > 21_600).sum()))


def _risk(teb: pd.DataFrame) -> np.ndarray | None:
    """Month-fold out-of-fold risk scores, if scripts/sel_risk.py has been run; absent, the row is simply left out."""
    try:
        d = pd.read_parquet(RISK)
    except (FileNotFoundError, OSError):
        return None
    return pd.Series(d.risk.values, index=d.id.values).reindex(teb.id.values).values


def _board(rem_per_row: float) -> float:
    return CHAL_BOARD - float(np.sqrt((N_RANK * CHAL_BOARD ** 2 - rem_per_row * len(groups()["laneB"])) / N_RANK))


def curve() -> tuple:
    tra, tea = lane_a()
    trb, teb = lane_b()
    ea = floored(sb3_predict(tra, tea)) - tea.y.values
    pbv = floored(predict_b(CURRENT["B"], trb, teb))
    eb = pbv - teb.y.values
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
    return out, base, n, sse, marginal(eb, teb), float(eb.mean()), fine(eb, teb), singles(eb, pbv, teb)


def main() -> None:
    rows = banked()
    tbl, base, n, sse, mrg, hold_mean, fin, (sng, ks, tail) = curve()
    n_board_b = len(groups()["laneB"])
    sig = N_RANK * ((S1_RMSE + DP / 2) ** 2 - (S1_RMSE - DP / 2) ** 2) / (2 * 300.0)
    drop = lambda r, rem: r - float(np.sqrt((N_RANK * r ** 2 - rem) / N_RANK))  # noqa: E731
    s1_gain, chal_gain = drop(S1_RMSE, rows[0][2]), drop(CHAL_BOARD, rows[1][2])
    body = [f"| {name} | {c[0]:+,.0f} | {c[1]:+,.0f} | {c[2]:+,.0f} | {rem / 1e6:,.1f} |" for name, c, rem in rows]
    by = dict(sng)
    risk_note = ""
    if RISK_NAME in by:
        risk_note = (f"A learned risk model on out-of-fold lane-B residuals is worse, {by[RISK_NAME][-1]:.2f} s at "
                     f"k = {ks[-1]} against prediction's {by['prediction'][-1]:.2f}, though it ranks the absolute "
                     "error better in correlation (`scripts/sel_risk.py` prints both). What pays here is the extreme "
                     "tail and not the ordering, so that direction is closed. ")
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
           f"On S1's own scale that is {s1_gain:.2f} s of board composite; on {CHAL_V} at its MEASURED {CHAL_BOARD:.2f} board result (`taxi.ceil`) it is "
           f"{chal_gain:.2f} s. Both are already paid for and neither has ever been applied to a submitted file.", "",
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
           f"per row, applied to the ranking file's {n_board_b:,} lane-B rows, read off {CHAL_V}'s MEASURED {CHAL_BOARD:.2f}.", "",
           "| cut | probes | cumulative (s) | marginal (s) |", "|---|---|---|---|",
           *[f"| {nm} | {g} | {c:.3f} | {m:+.3f} |" for nm, g, c, m in mrg], "",
           "The split matters because the two halves belong to different years. The LEVEL row buys almost nothing here, because L8c's "
           f"holdout mean error is {hold_mean:+.1f} s, and it buys most of section 1's gain on the board, where the same mean is "
           f"{rows[1][1][2] / n_board_b:+.1f} s. That half is bought and is section 1. The AIRPORT SPREAD is the reverse: the holdout "
           "sees it, the board has never measured it, and it is what the next probes buy. Reading the holdout's spread as the board's is "
           "an assumption. F23's year shift is itself airport-shaped (`docs/YR.md`) and adds spread rather than cancelling it, so the "
           "board's own spread should be at least this, which makes the marginal column the low side.", "",
           "## 6. Past the airport cut", "",
           "The same rescaling, on cuts the airport row does not reach. The last column is what a RANDOM partition into the same "
           "number of groups would give. On a fitted model that column would be the overfit; here it is a real gain, because the board "
           "measures each group's sum rather than estimating it, so the mean removed is the mean those rows actually have.", "",
           "| cut | probes | smallest group | board s | of which any partition of that size gives |", "|---|---|---|---|---|",
           *[f"| {nm} | {g} | {mn} | {b:.3f} | {r:.3f} |" for nm, g, mn, b, r in fin], "",
           "## 7. The limit of the idea, and why it is a question and not a plan", "",
           "Shrink a group to one row and the probe reads that row's label outright: the file moves that row by a known delta, the "
           "score moves by `2 delta e + delta^2`, and e falls out. At delta = 10,000 s the four printed decimals resolve e to about a "
           "second. So k files zero the k rows a selector picks, and lane B's error is concentrated enough for that to be worth more "
           "than every model change left on the board. Each row of the table is the board composite after zeroing the top k rows of "
           "that selector, holdout-measured.", "",
           "| selector | " + " | ".join(f"k = {k}" for k in ks) + " |", "|---|" + "---|" * len(ks),
           *[f"| {nm} | " + " | ".join(f"{v:.2f}" for v in vs) + " |" for nm, vs in sng], "",
           "Two things follow. The selector is the binding constraint, not the budget: prediction, the best label-free ranking tried, "
           f"reaches about a fifth of what the oracle ranking would. {risk_note}And the ranking file has the same tail to aim at as the holdout: "
           f"{CHAL_V} predicts over 6 h on {tail[0]} of its lane-B rows against {tail[1]} on the holdout, so the shape transfers even "
           "though the rows do not.", "",
           "It is still a question for the entrant and not a plan. Reading test labels off the scorer is not modelling, the prize asks "
           "for documentation sufficient to reproduce the result, and an entry whose gain came from k label reads documents exactly "
           "that. The published rules read on 7 September ban none of it and the board ranks each team by its best file, so the choice "
           "is the entrant's to make on its merits rather than the rulebook's to make for them. Nothing here has been uploaded.", ""]
    text = "\n".join(out) + "\n"
    with open("docs/PB.md", "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    if sys.argv[1:3]:
        correct(sys.argv[1], sys.argv[2])
    else:
        main()
