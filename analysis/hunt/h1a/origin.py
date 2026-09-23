"""Q1/Q3: where did the recorded off-block come from on every development-month departure with y > 1 h?
Run: PYTHONPATH=. python analysis/hunt/h1a/origin.py  (writes origin_dev.parquet in this folder)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import OUT, movements, v34_dev

TOL_SCHED = 10
TOL_STAMP = 60
DAY = 86400
LOCAL = {"LIRF": "Europe/Rome", "LFPG": "Europe/Paris", "LSZH": "Europe/Zurich", "LTFM": "Europe/Istanbul",
         "EGLL": "Europe/London", "EDDF": "Europe/Berlin", "EDDM": "Europe/Berlin", "EHAM": "Europe/Amsterdam",
         "LEMD": "Europe/Madrid", "LEBL": "Europe/Madrid"}


def local_day(t: pd.Series, apt: pd.Series) -> pd.Series:
    """Local calendar day number (days since epoch) of epoch-second times at each airport."""
    out = pd.Series(np.nan, index=t.index)
    for a, tz in LOCAL.items():
        m = (apt == a) & t.notna()
        if m.any():
            ts = pd.to_datetime(t[m], unit="s", utc=True).dt.tz_convert(tz).dt.tz_localize(None)
            out[m] = (ts.dt.normalize() - pd.Timestamp("1970-01-01")).dt.days
    return out


def classify(d: pd.DataFrame) -> pd.Series:
    b = d["BLOCK_TIME_UTC_mvt"]
    cls = pd.Series("other", index=d.index)
    k = np.round((d["MVT_TIME_UTC_mvt"] - b - 900) / DAY)
    rules = [
        ("sched", (b - d["SCHED_TIME_UTC_mvt"]).abs() <= TOL_SCHED),
        ("aobt", (b - d["AOBT_3_flt"]).abs() <= TOL_STAMP),
        ("eobt", (b - d["EOBT_1_flt"]).abs() <= TOL_STAMP),
        ("lobt", (b - d["LOBT_flt"]).abs() <= TOL_STAMP),
        ("iobt", (b - d["IOBT_flt"]).abs() <= TOL_STAMP),
        ("sched_kday", ((b - d["SCHED_TIME_UTC_mvt"]).abs() % DAY).pipe(lambda r: np.minimum(r, DAY - r)) <= TOL_SCHED),
        ("phys_kday", (k >= 1) & ((d["y"] - k * DAY).between(0, 3600))),
    ]
    for name, m in reversed(rules):
        cls[m.fillna(False)] = name
    return cls


def main() -> None:
    df = movements()
    d = df[(df["PHASE_mvt"] == "DEP") & df["dev"]].merge(v34_dev(), left_on="MVT_ID_mvt", right_on="id")
    d["lane"] = np.where(d["has_plan"], "A", "B")
    big = d[d["y"] > 3600].copy()
    big["cls"] = classify(big)
    big["e2"] = (big["y"] - big["p"]) ** 2
    big["dday_sched"] = local_day(big["MVT_TIME_UTC_mvt"] - 900, big["apt"]) - local_day(big["SCHED_TIME_UTC_mvt"], big["apt"])
    big["yb"] = pd.cut(big["y"], [3600, 7200, 14400, 43200, 1e9])
    t = big.groupby(["lane", "apt", "cls"]).agg(n=("y", "size"), sse=("e2", "sum")).reset_index()
    t["sse_B"] = t["sse"] / 1e9
    print(t.sort_values("sse", ascending=False).head(40).to_string())
    print(big.groupby(["lane", "yb", "cls"], observed=True).size().unstack(fill_value=0).to_string())
    big.drop(columns=["yb"]).to_parquet(OUT / "origin_dev.parquet")


if __name__ == "__main__":
    main()
