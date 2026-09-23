"""Does the inbound arrival on the same stand tell whether a departure's off-block will be a schedule copy?
For each departure, find the last arrival on the same stand at the same airport before takeoff (<= 36 h) and flag
whether that arrival's in-block equals its own schedule (an arrival-side copy). Label-free on rank rows.
Run: PYTHONPATH=. python analysis/hunt/h1a/inbound.py"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import movements

MAXGAP = 36 * 3600


def attach_inbound(df: pd.DataFrame) -> pd.DataFrame:
    """Departures with the previous same-stand arrival's stamps joined (merge_asof on takeoff time)."""
    dep = df[df["PHASE_mvt"] == "DEP"].copy()
    arr = df[(df["PHASE_mvt"] == "ARR") & df["STAND_mvt"].notna()].copy()
    arr = arr[["apt", "STAND_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "has_plan",
               "FLIGHT_mvt", "y"]].rename(columns=lambda c: c if c in ("apt", "STAND_mvt") else "in_" + c)
    arr = arr[arr["in_BLOCK_TIME_UTC_mvt"].notna()].sort_values("in_BLOCK_TIME_UTC_mvt")
    dep = dep[dep["STAND_mvt"].notna()].sort_values("MVT_TIME_UTC_mvt")
    out = pd.merge_asof(dep, arr, left_on="MVT_TIME_UTC_mvt", right_on="in_BLOCK_TIME_UTC_mvt",
                        by=["apt", "STAND_mvt"], direction="backward", tolerance=MAXGAP)
    out["in_copy"] = (out["in_BLOCK_TIME_UTC_mvt"] - out["in_SCHED_TIME_UTC_mvt"]).abs() <= 10
    out["in_gap_h"] = (out["MVT_TIME_UTC_mvt"] - out["in_BLOCK_TIME_UTC_mvt"]) / 3600
    out.loc[out["in_BLOCK_TIME_UTC_mvt"].isna(), "in_copy"] = np.nan
    return out


def main() -> None:
    df = movements()
    d = attach_inbound(df)
    d = d[d["dev"]].copy()
    d["lane"] = np.where(d["has_plan"], "A", "B")
    d["zs"] = d["MVT_TIME_UTC_mvt"] - d["SCHED_TIME_UTC_mvt"]
    d["copy"] = (d["y"] - d["zs"]).abs() <= 10
    pd.set_option("display.width", 250)
    big = d[d["zs"] > 3600]
    t = big.groupby(["lane", "apt", "in_copy"], dropna=False)["copy"].agg(["size", "mean"]).unstack("in_copy")
    print("departures with zs > 1 h: count and copy rate, by inbound-arrival copy flag\n", t.round(3).to_string())
    arr = df[(df["PHASE_mvt"] == "ARR") & df["dev"]]
    ac = ((arr["BLOCK_TIME_UTC_mvt"] - arr["SCHED_TIME_UTC_mvt"]).abs() <= 10).groupby([arr["apt"], arr["has_plan"]]).mean()
    print("\narrival in-block == schedule rate by apt/plan\n", ac.round(4).unstack().to_string())


if __name__ == "__main__":
    main()
