"""PV: the five raw columns no module has ever read, priced without a fit.

Diagnostic. No cell declared, no champion moves, no verdict decided. `taxi.data` reads 25 of the file's 30 columns; F30 closed
ARVT_3_flt and this closes the other four. Each has a twin the model already holds, so the only new information in it is the
disagreement between the flight plan and the movement record. Regenerate with `python -m taxi.pv`.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from taxi.champ import champion_holdout, lane_a
from taxi.data import HOLDOUT_MONTHS, _files, month_file
from taxi.verdict import paired_delta, rmse

DECILES = 10

RAW = """
SELECT MVT_ID_mvt AS id, FLIGHT_mvt, CALLSIGN_flt, ADES_mvt, ADES_FILED_flt,
       AIRCRAFT_TYPE_mvt, AIRCRAFT_TYPE_flt, FLIGHT_RULE_mvt, FLIGHT_RULE_flt
FROM read_parquet({files}) WHERE PHASE_mvt = 'DEP'"""


def raw(months) -> pd.DataFrame:
    return duckdb.connect().execute(RAW.format(files=_files([month_file(m) for m in months]))).df()


def flags(d: pd.DataFrame) -> dict:
    """Each flag is a disagreement between a plan field and its movement twin, which is the part no held column can express."""
    def ne(a: str, b: str) -> np.ndarray:
        return (d[a].notna() & d[b].notna() & (d[a] != d[b])).values
    return {"actype": ne("AIRCRAFT_TYPE_flt", "AIRCRAFT_TYPE_mvt"), "ades": ne("ADES_FILED_flt", "ADES_mvt"),
            "frule": ne("FLIGHT_RULE_flt", "FLIGHT_RULE_mvt"), "callsign": ne("CALLSIGN_flt", "FLIGHT_mvt")}


def recalibrated(err: np.ndarray, groups: list) -> np.ndarray:
    """The in-sample optimal per-group level, fit on the judged rows themselves: a CEILING for the level mechanism, not a fit."""
    e = err.copy()
    for sel in groups:
        if sel.sum() >= 30:
            e[sel] -= e[sel].mean()
    return e


def main() -> None:
    tra, tea = lane_a()
    pred = champion_holdout("MC2", tra, tea)
    r = raw(HOLDOUT_MONTHS).set_index("id").reindex(tea.id.values).reset_index()
    err, f = pred - tea.y.values, flags(r)
    days = (tea.t_off.values // 86400).astype(np.int64)
    q = pd.qcut(pred, DECILES, labels=False, duplicates="drop").astype(int)
    dec = [q == d for d in range(q.max() + 1)]
    base = paired_delta(err, recalibrated(err, dec), days).delta

    out = [f"# PV: the four unread plan columns, priced on the Jan + Jul 2025 development holdout", "",
           f"Regenerate with `python -m taxi.pv`. Champion MC2 on {len(tea):,} lane-A rows, {rmse(err):.4f} s. A flag is 1 where a "
           f"flight-plan field disagrees with the movement field it twins; that disagreement is the only part of these columns the "
           f"model cannot already compute. No fit is run and no cell is declared.", "",
           "## 1. Mass, difficulty and bias", "",
           "| flag | rows | share | MC2 on flagged | MC2 on rest | mean residual flagged | rest | not a function of `operator` |",
           "|---|---|---|---|---|---|---|---|"]
    for name, m in f.items():
        d = pd.DataFrame({"op": tea.operator.values, "m": m})
        maj = d.groupby("op", observed=True).m.transform(lambda s: s.mean() >= 0.5).values
        out.append(f"| {name} | {int(m.sum()):,} | {100 * m.mean():.3f}% | {rmse(err[m]):.1f} s | {rmse(err[~m]):.1f} s | "
                   f"{err[m].mean():+.1f} s | {err[~m].mean():+.1f} s | {int((m != maj).sum()):,} rows |")

    out += ["", "## 2. What a perfect recalibration on each flag is worth", "",
            f"The champion's residual is re-centred on the judged rows themselves, first inside {DECILES} deciles of its own "
            f"prediction and then inside each decile split by the flag. Fitting the correction on the rows it is judged on makes "
            f"this a ceiling for the level mechanism rather than a fit, and the decile-only row is its noise floor.", "",
            "| correction | ceiling [95%] |", "|---|---|",
            f"| deciles alone, no flag | {base:+.2f} s |"]
    for name, m in f.items():
        v = paired_delta(err, recalibrated(err, [s & g for s in dec for g in (m, ~m)]), days)
        out.append(f"| deciles x {name} | {v.delta:+.2f} [{v.lo:+.2f}, {v.hi:+.2f}] s |")

    out += ["", "## 3. Where the callsign flag looked like something", "",
            "It carries real mass, it is not a function of the operator, and MC2's error on it is 60 s below the rest of the lane, "
            "so it is the one flag that had to be taken apart rather than dismissed on rate. Inside a decile of MC2's own "
            "prediction the two groups differ in spread but not in level, and squared error pays for the level.", "",
            "| MC2 prediction decile | mean prediction | flagged rows | RMSE flagged | rest rows | RMSE rest | mean residual gap |",
            "|---|---|---|---|---|---|---|"]
    m = f["callsign"]
    for d, s in enumerate(dec):
        a, b = s & m, s & ~m
        out.append(f"| {d} | {pred[s].mean():.0f} s | {int(a.sum()):,} | {rmse(err[a]):.1f} s | {int(b.sum()):,} | "
                   f"{rmse(err[b]):.1f} s | {err[a].mean() - err[b].mean():+.1f} s |")
    out += ["", f"The mismatch is an encoding difference before it is anything else: `FLIGHT_mvt` carries the IATA commercial "
            f"number and `CALLSIGN_flt` the ICAO ATC callsign, so `LH1158` and `DLH73U` are the same flight. Its "
            f"{r.CALLSIGN_flt.str.slice(0, 3).nunique()} distinct prefixes sit against {tea.operator.nunique()} operators and "
            f"{tea.airline.nunique()} airline codes the model already reads.", ""]
    text = "\n".join(out) + "\n"
    with open("docs/PV.md", "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
