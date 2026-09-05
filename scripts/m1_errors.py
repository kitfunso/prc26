"""Diagnostic (protocol: not evidence): where the lane champions' holdout squared error sits, by airport, hour, tail and gap sign."""
from __future__ import annotations

import numpy as np
import pandas as pd

from taxi.baselines import b4
from taxi.data import HOLDOUT_MONTHS, TRAIN_MONTHS, load
from taxi.m1 import chosen_cells, predict, train

OUT = "data/holdout_pred.parquet"


def shares(df: pd.DataFrame, key: str, e: np.ndarray, top: int = 12) -> str:
    g = pd.DataFrame({"k": df[key].values, "se": e * e}).groupby("k", observed=True).se.agg(["count", "sum", "mean"])
    g["share"] = g["sum"] / g["sum"].sum()
    g["rmse"] = np.sqrt(g["mean"])
    return g.sort_values("share", ascending=False).head(top)[["count", "share", "rmse"]].to_string()


def main() -> None:
    tr, ho = load(TRAIN_MONTHS), load(HOLDOUT_MONTHS)
    cell, rounds = chosen_cells()["A"]
    pred = np.full(len(ho), np.nan)
    for lane in ("A", "B"):
        m = ho.has_fpl.values == (lane == "A")
        trl, hol = tr[tr.has_fpl == (lane == "A")].reset_index(drop=True), ho[m].reset_index(drop=True)
        pred[m] = predict(train(trl, lane, cell, rounds=rounds), hol, lane) if lane == "A" else b4(trl, hol, lane)
    ho = ho.assign(pred=pred, e=pred - ho.y.values)
    ho[["id", "apt", "month", "has_fpl", "y", "pred", "e"]].to_parquet(OUT)
    e = ho.e.values
    print(f"all rows {len(ho)}: rmse {np.sqrt((e**2).mean()):.1f}; lane A {np.sqrt((e[ho.has_fpl]**2).mean()):.1f}, "
          f"lane B {np.sqrt((e[~ho.has_fpl]**2).mean()):.1f}; lane B SSE share {(e[~ho.has_fpl]**2).sum() / (e**2).sum():.3f}")
    for lane in ("A", "B"):
        d = ho[ho.has_fpl == (lane == "A")]
        el = d.e.values
        ae = np.abs(el)
        print(f"\n== lane {lane}: {len(d)} rows, rmse {np.sqrt((el**2).mean()):.1f}, mean error {el.mean():+.1f}")
        for q in (0.9, 0.99, 0.999):
            k = int(len(d) * (1 - q))
            top = np.sort(el**2)[::-1][:k].sum() / (el**2).sum()
            print(f"  top {100 * (1 - q):.1f}% rows ({k}) carry {top:.3f} of SSE")
        for cut in (300, 600, 1800, 3600):
            print(f"  |e| > {cut}: {int((ae > cut).sum())} rows, SSE share {(el[ae > cut]**2).sum() / (el**2).sum():.3f}")
        print("  under-predicted (pred < y) SSE share:", f"{(el[el < 0]**2).sum() / (el**2).sum():.3f}")
        for key in ("apt", "hour", "wday", "runway"):
            if key in d.columns:
                print(f"  by {key}:\n" + "\n".join("    " + l for l in shares(d, key, el).splitlines()))
        if lane == "A":
            neg = d.z_aobt.values < 0
            print(f"  z_aobt < 0: {int(neg.sum())} rows, SSE share {(el[neg]**2).sum() / (el**2).sum():.3f}")


if __name__ == "__main__":
    main()
