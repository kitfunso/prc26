"""LIRF lane B: label-free features next to the label class (copy / physical / shift), development months only.
Run: PYTHONPATH=. python analysis/hunt/h1a/lirf_b.py"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from common import OUT, movements, v34_dev
from origin import DAY, local_day


def fmt(f: object) -> str:
    """Shape of the flight-number string: none, plain (AAA123), doubled prefix (ITYTY630, WMTMT4MV), other."""
    if not isinstance(f, str):
        return "none"
    if re.fullmatch(r"[A-Z]{3}\d[0-9A-Z]{0,3}", f):
        return "plain"
    if len(f) >= 5 and f[3:5] == f[1:3]:
        return "dbl2"
    if len(f) >= 4 and f[3] == f[2]:
        return "dbl1"
    return "other"


def frame(apt: str = "LIRF", lane_b: bool = True) -> pd.DataFrame:
    df = movements()
    dep = df[df["PHASE_mvt"] == "DEP"]
    d = dep[dep["dev"] & (dep["apt"] == apt) & (dep["has_plan"] != lane_b)].merge(
        v34_dev(), left_on="MVT_ID_mvt", right_on="id")
    d["zs"] = d["MVT_TIME_UTC_mvt"] - d["SCHED_TIME_UTC_mvt"]
    d["cls"] = np.where((d["y"] - d["zs"]).abs() <= 10, "copy", np.where(d["y"] > 43200, "shift", "phys"))
    d["fmt"] = d["FLIGHT_mvt"].map(fmt)
    d["pfx"] = d["FLIGHT_mvt"].str[:3]
    d["stand1"] = d["STAND_mvt"].str[:1]
    d["lh_mvt"] = pd.to_datetime(d["MVT_TIME_UTC_mvt"], unit="s", utc=True).dt.tz_convert("Europe/Rome").dt.hour
    d["lh_std"] = pd.to_datetime(d["SCHED_TIME_UTC_mvt"], unit="s", utc=True).dt.tz_convert("Europe/Rome").dt.hour
    d["jit"] = ((d["MVT_TIME_UTC_mvt"] + 30) % 60) - 30
    d["dday"] = local_day(d["MVT_TIME_UTC_mvt"] - 900, d["apt"]) - local_day(d["SCHED_TIME_UTC_mvt"], d["apt"])
    d["zsb"] = pd.cut(d["zs"] / 3600, [-99, 0, 0.5, 1, 2, 4, 12, 999]).astype(str)
    return d


def main() -> None:
    d = frame()
    pd.set_option("display.width", 250)
    print(d.groupby(["zsb", "cls"]).size().unstack(fill_value=0).to_string())
    for c in ["fmt", "stand1", "lh_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt", "FLIGHT_RULE_mvt", "month"]:
        t = d[d["zs"] > 3600].groupby(c, dropna=False)["cls"].value_counts().unstack(fill_value=0)
        print(f"\n{c} (zs > 1 h)\n", t.to_string())
    d.to_parquet(OUT / "lirf_b_dev.parquet")


if __name__ == "__main__":
    main()
