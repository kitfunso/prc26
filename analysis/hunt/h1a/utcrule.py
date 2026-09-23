"""Test a third date rule for day-shifted off-blocks: recorded BLOCK = the schedule's UTC date + the actual off-block's
UTC clock time, so k = UTC date(actual) - UTC date(schedule). Compared with the local-date and nearest rules on every
non-copy development departure (actual = BLOCK + k days, k from y). Run: PYTHONPATH=. python analysis/hunt/h1a/utcrule.py"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import movements
from origin import DAY, local_day


def main() -> None:
    df = movements()
    d = df[(df["PHASE_mvt"] == "DEP") & df["dev"]].copy()
    d["lane"] = np.where(d["has_plan"], "A", "B")
    d["zs"] = d["MVT_TIME_UTC_mvt"] - d["SCHED_TIME_UTC_mvt"]
    d = d[(d["y"] - d["zs"]).abs() > 10]
    d["k"] = np.round((d["y"] - 1200) / DAY)
    d["actual"] = d["BLOCK_TIME_UTC_mvt"] + d["k"] * DAY
    d["dutc"] = np.floor(d["actual"] / DAY) - np.floor(d["SCHED_TIME_UTC_mvt"] / DAY)
    d["dloc"] = local_day(d["actual"], d["apt"]) - local_day(d["SCHED_TIME_UTC_mvt"], d["apt"])
    d["near"] = np.round((d["actual"] - d["SCHED_TIME_UTC_mvt"]) / DAY)
    pd.set_option("display.width", 250)
    s = d[(d["dutc"] != 0) | (d["dloc"] != 0) | (d["near"] != 0) | (d["k"] != 0)]
    for lane in ("B", "A"):
        t = s[s["lane"] == lane].groupby(["apt", "dutc", "dloc", "near", "k"]).size()
        print(f"lane {lane}: non-copy rows where any rule or the label says a day moved\n{t.to_string()}\n")
    b = d[(d["lane"] == "B") & (d["apt"] == "LIRF")]
    for rule in ("dutc", "dloc", "near"):
        print(f"LIRF lane B, rule {rule}: agrees with label on {(b[rule] == b['k']).sum()} of {len(b)} non-copy rows")
    x = b[(b["dutc"] != b["k"]) | (b["k"] != 0)]
    x = x.assign(sched=pd.to_datetime(x["SCHED_TIME_UTC_mvt"], unit="s"), act=pd.to_datetime(x["actual"], unit="s"))
    print(x[["MVT_ID_mvt", "FLIGHT_mvt", "sched", "act", "y", "k", "dutc", "dloc", "near"]].to_string())


if __name__ == "__main__":
    main()
