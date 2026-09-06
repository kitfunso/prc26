"""TC: what is the label's own clock made of, and does any of it pay?

Diagnostic. No cell declared, no champion moves, no verdict decided. The build carries `aobt_sec`, `aobt_m5`,
`eobt_m5` and `lobt_sec`, so rounding in the FLIGHT-PLAN clocks was noticed early; nothing ever asked the same
question of the movement clock that defines the target. Regenerate with `python -m taxi.tc`.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from taxi.champ import lane_a
from taxi.data import month_file

MONTHS = tuple(range(1, 13))
COPIES = (("y = z_aobt (AOBT copy, c = 0)", "z_aobt"), ("y = z_sched (schedule copy)", "z_sched"),
          ("y = z_eobt (EOBT copy)", "z_eobt"), ("y = z_lobt (LOBT copy)", "z_lobt"))


def grid() -> pd.DataFrame:
    """Per airport minute-rounding of each clock. Two units, because one rate alone cannot tell rounding from a coarse grid."""
    paths = [month_file(m) for m in MONTHS]
    d = duckdb.sql(f"""SELECT ADEP_mvt apt, count(*) n,
      100.0*avg((epoch(BLOCK_TIME_UTC_mvt) % 60 = 0)::INT) blk_min,
      100.0*avg((epoch(BLOCK_TIME_UTC_mvt) % 5 = 0)::INT) blk_5s,
      100.0*avg((epoch(AOBT_3_flt) % 60 = 0)::INT) aobt_min
    FROM read_parquet({paths}) WHERE PHASE_mvt='DEP' AND TAXITIME_SEC_mvt IS NOT NULL
    GROUP BY 1 ORDER BY blk_min DESC""").df()
    # A share q rounded to the minute and the rest uniform predicts both rates; a coarse grid would not fit both.
    d["rounded"] = 100 * (d.blk_min / 100 - 1 / 60) / (1 - 1 / 60)
    d["pred_5s"] = d.rounded + (100 - d.rounded) * 0.2
    return d


def copies() -> pd.DataFrame:
    paths = [month_file(m) for m in MONTHS]
    return duckdb.sql(f"""SELECT (AOBT_3_flt IS NOT NULL) lane_a, count(*) n,
      100.0*avg((BLOCK_TIME_UTC_mvt = SCHED_TIME_UTC_mvt)::INT) sched,
      100.0*avg((BLOCK_TIME_UTC_mvt = EOBT_1_flt)::INT) eobt,
      100.0*avg((BLOCK_TIME_UTC_mvt = LOBT_flt)::INT) lobt,
      100.0*avg((BLOCK_TIME_UTC_mvt = AOBT_3_flt)::INT) aobt
    FROM read_parquet({paths}) WHERE PHASE_mvt='DEP' AND TAXITIME_SEC_mvt IS NOT NULL
    GROUP BY 1 ORDER BY 1""").df()


def main() -> None:
    g, cp = grid(), copies()
    _, tea = lane_a()
    tea = tea.reset_index(drop=True)
    mc2 = pd.read_parquet("data/cache/mc2_holdout.parquet")
    pred, y = mc2.pred.values, tea.y.values
    n = len(y)
    base = float(np.sqrt(np.mean((pred - y) ** 2)))
    tot = float(np.sum((pred - y) ** 2))

    rows, cum = [], np.zeros(n, bool)
    for nm, col in COPIES:
        m = np.isclose(y, tea[col].values)
        cum |= m
        sse = float(np.sum((pred[m] - y[m]) ** 2))
        rows.append(f"| {nm} | {m.sum():,} | {100 * m.mean():.3f}% | {np.sqrt(sse / max(m.sum(), 1)):,.0f} | "
                    f"{100 * sse / tot:.2f}% | {base - np.sqrt((tot - sse) / n):+.2f} |")
    sched = np.isclose(y, tea.z_sched.values)
    sched_rmse = float(np.sqrt(np.mean((pred[sched] - y[sched]) ** 2)))
    sse_all = float(np.sum((pred[cum] - y[cum]) ** 2))
    oracle = base - np.sqrt((tot - sse_all) / n)
    rows.append(f"| **any of the four** | {cum.sum():,} | {100 * cum.mean():.3f}% | "
                f"{np.sqrt(sse_all / cum.sum()):,.0f} | {100 * sse_all / tot:.2f}% | {oracle:+.2f} |")

    fmt = lambda v: f"{v:7.3f}"
    out = [
        "# TC: the label's own clock, and whether any of its structure pays",
        "",
        "Regenerate with `python -m taxi.tc`. Diagnostic: no cell declared, no champion moves, no verdict decided.",
        f"MC2 holdout RMSE {base:.4f} s on {n:,} lane-A rows, against the 2.0 s improvement floor.",
        "",
        "## 1. The recording grid",
        "",
        "The target is `y = takeoff - BLOCK_TIME` and the takeoff time is given on every ranking row, so any lattice the",
        "block time is recorded on is a lattice the answer sits on. That would be free accuracy, needing no model.",
        "",
        g.to_string(index=False, float_format=fmt),
        "",
        "`AOBT_3` is written to the whole minute nearly everywhere, which the build already reads through `aobt_sec` and",
        "`aobt_m5`. The block time is not. Reading its 8.3% minute rate at seven airports as a 5-second recording grid was",
        "wrong, and the second unit is what caught it: a 5-second grid would put `blk_5s` at 100% and it is 25%. The rate",
        "that does fit both columns is a MINORITY of rounded rows, a share q written to the minute and the rest recorded",
        f"to the second, which implies q = {g.rounded.max():.1f}% at the top airport and predicts `blk_5s` of {g.pred_5s.max():.1f}% against the",
        f"{g.blk_5s.max():.1f}% observed. The three airports at the bottom of the table show 1.7% and 20%, which is chance in both",
        "columns: no rounding at all.",
        "",
        "Either way the idea is closed by arithmetic, and by a wider margin than the mistaken reading would have given.",
        "A lattice only buys accuracy when the residual is comparable to its spacing. Snapping onto a 60 s lattice removes",
        "at most its quantisation variance, 3600/12 = 300 against MC2's 54,162, which is 0.65 s on a row known to be",
        f"rounded; at {g.rounded.max():.1f}% of rows at seven of ten airports, and with no way to identify which rows those are, the",
        "expected return is about 0.04 s. Membership is the binding problem, not the size of the lattice.",
        "",
        "## 2. Is the block time a copy of another clock rather than a measurement?",
        "",
        "The lane-B champion is built around exactly this: L4 to L8 model rows whose block time was copied from the",
        "schedule, and L8c is a CatBoost classifier choosing between the copy branch and the regression. F28 then showed",
        "why the shape matters, that a linear branch emits `y = z_sched` exactly while a tree emits piecewise constants.",
        "Lane A carries 51.4% of board squared error and has never been asked the question.",
        "",
        cp.to_string(index=False, float_format=fmt),
        "",
        "## 3. The oracle: perfect knowledge of which lane-A rows are copies, and the exact value on them",
        "",
        "| copy branch | rows | share | MC2 RMSE on them | share of lane SSE | oracle gain (s) |",
        "|---|---|---|---|---|---|",
        *rows,
        "",
        "## 4. Answer: both channels are closed, and the second is the one worth recording",
        "",
        f"Copy rows are genuinely bad rows. MC2 is at {sched_rmse:.0f} s on schedule copies against {base:.0f} s over the lane, so they are",
        f"mispriced by {sched_rmse / base - 1:.0%}. They are also rare, {100 * cum.mean():.3f}% of the lane, and rarity wins: perfect classification plus",
        f"the exact value on every one of the four branches is worth {oracle:+.2f} s, under the 2.0 s floor. Lane B's classifier is",
        "nowhere near perfect, so a port of it to lane A would collect a fraction of a number that is already too small.",
        "",
        "This is the closure that matters, because F28's mechanism makes the port look obvious. A tree cannot emit",
        "`y = z_sched` exactly in lane A either, and the reflex is to give lane A the branch lane B has. The oracle says",
        "not to: the mechanism is real in both lanes and the mass is not. Lane B's copies carry a lane where 13 rows hold",
        f"67% of the squared error; lane A's carry {100 * sse_all / tot:.2f}% of theirs.",
        "",
        "Consequence: do not declare a lane-A copy branch, a copy classifier, or a mixture ported from L4 to L8c, and do",
        "not treat the label's recording grid as free accuracy.",
        "",
    ]
    text = "\n".join(out) + "\n"
    with open("docs/TC.md", "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
