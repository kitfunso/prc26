"""Test the day-shift mechanism: recorded off-block = schedule's LOCAL date + actual off-block's local clock time.
Counts label classes by the local-date gap between (takeoff - 15 min) and the schedule, per airport and lane.
Run: PYTHONPATH=. python analysis/hunt/h1a/dayshift.py"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import movements, v34_dev
from origin import DAY, local_day


def label_class(y: pd.Series, zs: pd.Series) -> pd.Series:
    k = np.floor(y / DAY)
    c = pd.Series("normal", index=y.index)
    c[(k >= 1) & (y - k * DAY <= 3600)] = "shift" + k.clip(upper=3).astype("Int64").astype(str)
    c[(y - zs).abs() <= 10] = "copy"
    c[(c == "normal") & (y > 3600)] = "big_other"
    return c


def main() -> None:
    df = movements()
    d = df[(df["PHASE_mvt"] == "DEP") & df["dev"]].merge(v34_dev(), left_on="MVT_ID_mvt", right_on="id")
    d["lane"] = np.where(d["has_plan"], "A", "B")
    d["zs"] = d["MVT_TIME_UTC_mvt"] - d["SCHED_TIME_UTC_mvt"]
    d["dday"] = (local_day(d["MVT_TIME_UTC_mvt"] - 900, d["apt"]) - local_day(d["SCHED_TIME_UTC_mvt"], d["apt"])).clip(-1, 3)
    d["cls"] = label_class(d["y"], d["zs"])
    d["e2"] = (d["y"] - d["p"]) ** 2
    pd.set_option("display.width", 250)
    for lane in ("B", "A"):
        s = d[d["lane"] == lane]
        t = s.groupby(["apt", "dday", "cls"]).size().unstack(fill_value=0)
        print(f"lane {lane}: label class by local-date gap (takeoff-15min vs schedule)\n", t.to_string())
    s = d[(d["dday"] >= 1)]
    print(s.groupby(["lane", "apt", "cls"]).agg(n=("y", "size"), sse=("e2", "sum")).to_string())


if __name__ == "__main__":
    main()
