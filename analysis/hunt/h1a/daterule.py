"""Which rule sets the recorded DATE of a physical off-block? For every non-copy development departure, rebuild the
true pushback (BLOCK + k days, k from y), then compare k with two candidate rules:
  date rule    k = local date(actual) - local date(schedule)
  nearest rule k = round((actual - schedule) / 1 day)
Run: PYTHONPATH=. python analysis/hunt/h1a/daterule.py"""
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
    d["dloc"] = local_day(d["actual"], d["apt"]) - local_day(d["SCHED_TIME_UTC_mvt"], d["apt"])
    d["near"] = np.round((d["actual"] - d["SCHED_TIME_UTC_mvt"]) / DAY)
    d["delay_h"] = pd.cut((d["actual"] - d["SCHED_TIME_UTC_mvt"]) / 3600, [-48, -12, 0, 6, 12, 18, 24, 48, 999])
    pd.set_option("display.width", 250)
    print("k by lane/apt (non-copy rows):\n", d.groupby(["lane", "apt", "k"]).size().unstack(fill_value=0).to_string())
    s = d[(d["dloc"] != 0) | (d["near"] != 0) | (d["k"] != 0)]
    print("\nrows where any rule or the label says shift:\n",
          s.groupby(["lane", "apt", "dloc", "near", "k"]).size().to_string())
    b = d[(d["lane"] == "B") & (d["apt"] == "LIRF")]
    print("\nLIRF lane B non-copy: k by delay bin and dloc\n",
          b.groupby(["delay_h", "dloc", "k"], observed=True).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
