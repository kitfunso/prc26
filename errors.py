"""Error analysis of a config's OOF and holdout predictions by airport, lane, hour and off-block mechanism. `python errors.py NAME`"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from data import CACHE, rmse
from features import build
from train import HOLDOUT

pd.set_option("display.width", 250)


def table(d: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    d = d.assign(se=(d["y"] - d["p"]) ** 2)
    g = d.groupby(by, observed=True)
    out = pd.DataFrame({"n": g.size(), "rmse": np.sqrt(g["se"].mean()), "bias": g.apply(lambda x: (x["p"] - x["y"]).mean(), include_groups=False),
                        "sse%": 100 * g["se"].sum() / d["se"].sum()})
    return out.round(1)


def mechanism(d: pd.DataFrame) -> pd.Series:
    m = pd.Series("physics", index=d.index)
    m[(d["y"] - d["z_sched"]).abs() < 60] = "sched_copy"
    m[(d["y"] - d["z_eobt"]).abs() < 60] = "eobt_copy"
    m[(d["y"] - d["z_aobt"]).abs() < 60] = "aobt_copy"
    m[d["y"] > 72000] = "day_shift"
    return m


def main(name: str) -> None:
    f = build()
    f = f[f["split"] == "train"]
    for kind, months in (("oof", None), ("holdout", HOLDOUT)):
        d = f[f["month"].isin(months)] if months else f[~f["month"].isin(HOLDOUT)]
        d = d.merge(pd.read_parquet(CACHE / f"{name}_{kind}.parquet").rename(columns={"oof": "p"}), on="id")
        d["p"] = np.maximum(d["p"], 0)
        d["lane"] = d["has_plan"].map({1: "A", 0: "B"}); d["mech"] = mechanism(d)
        print(f"\n=== {name} {kind}: rmse {rmse(d['y'], d['p']):.2f}, n={len(d):,}")
        print(table(d, ["lane", "apt"]).to_string())
        print(table(d, ["lane", "mech"]).to_string())
        print(table(d, ["lane", pd.cut(d["y"], [-1, 600, 1200, 1800, 3600, 7200, 21600, 72000, 1e9])]).to_string())
        b = d[d["lane"] == "B"]
        print("lane B by airport x mech:\n", table(b, ["apt", "mech"]).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
