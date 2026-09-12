"""Holdout error breakdown for a cached blend: RMSE and squared-error share by airport, lane and month. `python errors.py CACHE_NAME`."""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from data import CACHE, rmse
from features import build
from train import HOLDOUT


def main(name: str) -> None:
    f = build()
    h = f[(f["split"] == "train") & f["month"].isin(HOLDOUT)].merge(pd.read_parquet(CACHE / f"{name}_holdout.parquet"), on="id")
    h["se"] = (h["y"] - h["p"]) ** 2
    tot = h["se"].sum()
    for key in (["apt"], ["apt", "has_plan"], ["month", "has_plan"]):
        g = h.groupby(key).agg(n=("se", "size"), rmse=("se", lambda s: float(np.sqrt(s.mean()))), share=("se", lambda s: float(s.sum() / tot)),
                               bias=("y", "mean"))
        g["bias"] = h.groupby(key).apply(lambda d: float((d["p"] - d["y"]).mean()))
        print(g.round(2).to_string(), "\n")
    a = h[h["has_plan"] == 1]
    for q in (600, 1800, 3600):
        k = (a["y"] - a["z_aobt"]).abs() > q
        print(f"laneA |c|>{q}: n={int(k.sum())} rmse={rmse(a['y'][k], a['p'][k]):.1f} share={a['se'][k].sum() / tot:.3f} bias={(a['p'][k] - a['y'][k]).mean():+.1f}")
    print(f"total rmse {rmse(h['y'], h['p']):.2f}")


if __name__ == "__main__":
    main(sys.argv[1])
