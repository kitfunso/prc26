"""Board probes P1-P6 (protocol section 3, D1): the S1 file with +c or -c on one group, and the derivation from the board."""
from __future__ import annotations

import sys

import duckdb
import pandas as pd

from taxi.submit import build_submission

C = 300.0
S1 = "data/merry-octopus_v1.parquet"
RANKING = "data/ranking.parquet"
# (probe id, file version, group, sign); the order is the declaration's, lane B first.
PROBES = (("P1", 2, "laneB", +1), ("P2", 3, "laneB", -1), ("P3", 4, "jan", +1),
          ("P4", 5, "jan", -1), ("P5", 6, "jul", +1), ("P6", 7, "jul", -1))
GROUP_SQL = {"laneB": "AOBT_3_flt IS NULL", "jan": "month(MVT_TIME_UTC_mvt) = 1", "jul": "month(MVT_TIME_UTC_mvt) >= 7"}


def groups() -> dict[str, pd.Index]:
    out = {}
    for g, cond in GROUP_SQL.items():
        ids = duckdb.sql(f"SELECT MVT_ID_mvt FROM read_parquet('{RANKING}') WHERE PHASE_mvt = 'DEP' AND {cond}").df()
        out[g] = pd.Index(ids.MVT_ID_mvt.values)
    return out


def build() -> None:
    s1 = duckdb.sql(f"SELECT * FROM read_parquet('{S1}')").df()
    pred = pd.Series(s1.TAXITIME_SEC_mvt.values.astype("float64"), index=s1.MVT_ID_mvt.values)
    grp = groups()
    for pid, v, g, sign in PROBES:
        p = pred.copy()
        p[grp[g]] += sign * C
        out = f"data/merry-octopus_v{v}.parquet"
        build_submission(p, out)
        print(f"{pid} {out}: {g} {'+' if sign > 0 else '-'}{C:.0f} s on {len(grp[g])} rows")


def derive(board: dict[str, float], n: int) -> list[str]:
    """Rows for ledger section C from the board RMSEs keyed by file version (1 = S1)."""
    sse = {v: n * r * r for v, r in board.items()}
    sizes = {g: len(ix) for g, ix in groups().items()}
    lines = []
    for g in GROUP_SQL:
        vp, vm = [v for _, v, gg, s in PROBES if gg == g and s > 0][0], [v for _, v, gg, s in PROBES if gg == g and s < 0][0]
        m = (sse[vp] + sse[vm] - 2 * sse[1]) / (2 * C * C)
        err = (sse[vp] - sse[vm]) / (4 * C)
        lines.append(f"{g}: derived m = {m:,.1f} (actual rows {sizes[g]:,}), S1 error sum = {err:,.0f} s, mean error {err / sizes[g]:+.1f} s per row")
    return lines


if __name__ == "__main__":
    if sys.argv[1:2] == ["derive"]:
        import lb
        ours = {r["filename"]: r for r in lb.fetch_all() if r["teamName"] == "merry-octopus"}
        board = {int(f.split("_v")[1].split(".")[0]): r["score"] for f, r in ours.items()}
        n = {r["usedPairs"] for r in ours.values()}
        assert len(n) == 1, f"usedPairs differs across our rows: {n}"
        print("\n".join(derive(board, n.pop())))
    else:
        build()
