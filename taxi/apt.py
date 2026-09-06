"""APT: per-airport structure diagnostic on MC2's out-of-fold lane-A predictions. Never decides a verdict (protocol section 2)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from taxi.champ import champion_oof, lane_a
from taxi.data import CV_FOLDS

MIN_SHIFT, MIN_LINE = 50, 200


def recal(oof: np.ndarray, y: np.ndarray, mon: np.ndarray, keys: np.ndarray, kind: str) -> np.ndarray:
    """Cross-fitted recalibration: a fold's rows are corrected by a per-key shift or line fitted on the other folds' OOF rows only."""
    out = oof.copy()
    for fold in CV_FOLDS:
        m = np.isin(mon, fold)
        f = pd.DataFrame({"k": keys[~m], "o": oof[~m], "y": y[~m]})
        if kind == "shift":
            adj = f.assign(d=f.y - f.o).groupby("k").d.agg(["mean", "size"])
            out[m] = oof[m] + adj["mean"].where(adj["size"] >= MIN_SHIFT, 0.0).reindex(pd.Index(keys[m])).fillna(0.0).values
        else:
            par = {k: np.polyfit(s.o.values, s.y.values, 1) for k, s in f.groupby("k") if len(s) >= MIN_LINE and s.o.std() > 0}
            a = np.array([par.get(k, (1.0, 0.0))[0] for k in keys[m]])
            b = np.array([par.get(k, (1.0, 0.0))[1] for k in keys[m]])
            out[m] = a * oof[m] + b
    return out


def main() -> None:
    tra, _ = lane_a()
    oof = champion_oof("MC2", tra)
    y, apt, mon = tra.y.values, tra.apt.astype(str).values, tra.month.values
    e = oof - y
    rmse = lambda p: float(np.sqrt(((p - y) ** 2).mean()))
    print("lane A train rows", len(tra), "OOF RMSE", round(rmse(oof), 2))

    g = pd.DataFrame({"apt": apt, "sq": e**2, "e": e})
    t = g.groupby("apt").agg(rows=("e", "size"), rmse=("sq", lambda s: float(np.sqrt(s.mean()))), bias=("e", "mean"),
                             share=("sq", lambda s: s.sum() / (e**2).sum()))
    print("\n--- per airport (OOF) ---")
    print(t.assign(rmse=t.rmse.round(1), bias=t.bias.round(1), share=(t.share * 100).round(1)).sort_values("share", ascending=False).to_string())

    print("\n--- cross-fitted corrections on MC2's OOF (RMSE, lower is better) ---")
    print(f"{'none':28s} {rmse(oof):8.2f}")
    for name, keys in (("airport", apt), ("airport x runway", apt + "|" + tra.runway.astype(str).values),
                       ("airport x hour", apt + "|" + tra.hour.astype(str).values),
                       ("airport x stand", apt + "|" + tra.stand.astype(str).values)):
        for kind in ("shift", "line"):
            print(f"{name + ' ' + kind:28s} {rmse(recal(oof, y, mon, keys, kind)):8.2f}")

    print("\n--- error concentration, lane A OOF ---")
    s = np.sort(e**2)[::-1]
    for n in (10, 100, 1000, 10000):
        print(f"top {n:6d} rows: {s[:n].sum() / s.sum() * 100:5.2f}% of squared error")


if __name__ == "__main__":
    main()
