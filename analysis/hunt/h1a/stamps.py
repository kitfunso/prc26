"""Q1/Q3: is the off-block of a big non-copy development departure (y > 1 h) a time stamp taken from ANOTHER row?
For each key (same flight number, same FLIGHT_ID, same stand, same airline prefix) find the nearest MVT / BLOCK / SCHED
stamp of any other movement at the airport and count hits within 60 s. The null is the same search at BLOCK + 10 min.
The target rows' own stamps are dropped from the table (merge_asof returns only the nearest stamp).
Run: PYTHONPATH=. python analysis/hunt/h1a/stamps.py"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import movements
from q4_decomp import label_class

TOL = 60
KEYS = {"flight": "FLIGHT_mvt", "flight_id": "FLIGHT_ID_mvt", "stand": "STAND_mvt", "prefix": "pfx"}


def stamp_table(df: pd.DataFrame, key: str) -> pd.DataFrame:
    cols = ["MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"]
    s = df[["apt", key, "MVT_ID_mvt"] + cols].dropna(subset=[key]).melt(
        id_vars=["apt", key, "MVT_ID_mvt"], value_vars=cols, var_name="src", value_name="t").dropna(subset=["t"])
    return s.rename(columns={"MVT_ID_mvt": "src_id"}).sort_values("t")


def nearest_hit(tgt: pd.DataFrame, st: pd.DataFrame, key: str, offset: float) -> pd.Series:
    """True where some OTHER row with the same key has a stamp within TOL of BLOCK + offset."""
    q = tgt[["MVT_ID_mvt", "apt", key]].assign(t=tgt["BLOCK_TIME_UTC_mvt"] + offset).dropna(subset=[key])
    hit = pd.Series(False, index=tgt["MVT_ID_mvt"].values)
    for side in ("backward", "forward"):
        m = pd.merge_asof(q.sort_values("t"), st, on="t", by=["apt", key], direction=side, allow_exact_matches=True,
                          tolerance=TOL, suffixes=("", "_s"))
        m = m[m["src_id"].notna() & (m["src_id"] != m["MVT_ID_mvt"])]
        hit[hit.index.isin(m["MVT_ID_mvt"])] = True
    return hit


def main() -> None:
    df = movements()
    df["pfx"] = df["FLIGHT_mvt"].str[:3]
    d = df[(df["PHASE_mvt"] == "DEP") & df["dev"]].copy()
    d["zs"] = d["MVT_TIME_UTC_mvt"] - d["SCHED_TIME_UTC_mvt"]
    d["cls"], _ = label_class(d["y"], d["zs"])
    tgt = d[(d["cls"] == "phys") & (d["y"] > 3600)].copy()
    tgt["lane"] = np.where(tgt["has_plan"], "A", "B")
    print(f"targets: {len(tgt)} physical-class dev departures with y > 1 h")
    res = {}
    for name, key in KEYS.items():
        st = stamp_table(df, key)
        st = st[~st["src_id"].isin(tgt["MVT_ID_mvt"])]
        for off, tag in ((0.0, "hit"), (600.0, "null")):
            res[(name, tag)] = nearest_hit(tgt, st, key, off).reindex(tgt["MVT_ID_mvt"].values).values
    r = pd.DataFrame({f"{a}_{b}": v for (a, b), v in res.items()})
    r[["lane", "apt"]] = tgt[["lane", "apt"]].values
    pd.set_option("display.width", 250)
    print(r.groupby(["lane", "apt"]).agg(["sum"]).droplevel(1, axis=1).assign(n=r.groupby(["lane", "apt"]).size()).to_string())


if __name__ == "__main__":
    main()
