"""TAIL: what carries the holdout composite, and what the extreme rows can still buy. Diagnostic, never a verdict (protocol section 2)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from taxi.champ import predict_b
from taxi.data import HOLDOUT_MONTHS, TRAIN_MONTHS, load
from taxi.verdict import level_ci

LANE_A_CACHE = "data/cache/mc2_holdout.parquet"
BANDS = ((0, 1e3), (1e3, 3e3), (3e3, 1e4), (1e4, 3e4), (3e4, 1e9))
FAR = 43200.0


def frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Holdout rows of both lanes with the current champions' floored predictions, and the lane-B training rows."""
    tr, te = load(TRAIN_MONTHS), load(HOLDOUT_MONTHS)
    trb, teb = tr[~tr.has_fpl].reset_index(drop=True), te[~te.has_fpl].reset_index(drop=True)
    a = te[te.has_fpl].reset_index(drop=True).merge(pd.read_parquet(LANE_A_CACHE)[["id", "pred"]], on="id", how="inner")
    b = teb.assign(pred=predict_b("L8c", trb, teb))
    out = pd.concat([a.assign(lane="A"), b.assign(lane="B")], ignore_index=True)
    out["pred"] = out.pred.clip(lower=0.0)
    return out, trb


def main() -> None:
    both, trb = frames()
    e = both.pred.values - both.y.values
    n, tot = len(both), float((e**2).sum())
    print(f"holdout rows {n}  composite {np.sqrt(tot/n):.4f} s")

    for lane, name in (("A", "MC2"), ("B", "L8c")):
        s = both[both.lane == lane]
        pt, sd, lo, hi = level_ci(s.pred.values - s.y.values, s.day.values)
        print(f"  lane {lane} {name:4s}: {pt:9.2f} s  bootstrap SD {sd:7.2f}  95% [{lo:8.2f}, {hi:8.2f}] on {len(s)} rows")
    pt, sd, lo, hi = level_ci(e, both.day.values)
    print(f"  composite  : {pt:9.2f} s  bootstrap SD {sd:7.2f}  95% [{lo:8.2f}, {hi:8.2f}]")

    print("\n--- squared-error share by |error| band ---")
    ae = np.abs(e)
    for lo_b, hi_b in BANDS:
        m = (ae >= lo_b) & (ae < hi_b)
        print(f"  |err| {lo_b:8.0f}-{hi_b:10.0f}: {int(m.sum()):6d} rows  {float((e[m]**2).sum())/tot*100:5.1f}%")

    print("\n--- every holdout row with |error| > 10000 s ---")
    big = both[ae > 1e4].assign(e2=e[ae > 1e4] ** 2).sort_values("e2", ascending=False)
    print(big[["lane", "apt", "month", "y", "pred", "z_sched", "notyp"]].to_string(index=False, float_format=lambda v: f"{v:.0f}"))

    print("\n--- far band (lane-B holdout rows with z_sched above 12 h) against the best constant mixture weight ---")
    b = both[both.lane == "B"]
    y, zs, pb = b.y.values, b.z_sched.values, b.pred.values
    far, ftr = zs > FAR, trb[trb.z_sched > FAR]
    cp = float((ftr.y - ftr.z_sched).abs().le(600).mean())
    print(f"  {int(far.sum())} holdout rows; training copy share {cp:.3f} on {len(ftr)} rows; current RMSE {np.sqrt(((pb[far]-y[far])**2).mean()):.0f} s")
    grid = [(float(((w * zs[far] + (1 - w) * ftr.y.mean() - y[far]) ** 2).sum()), w) for w in np.arange(0.0, 1.01, 0.05)]
    s, w = min(grid)
    print(f"  best constant weight w={w:.2f} -> RMSE {np.sqrt(s/far.sum()):.0f} s; the per-row P(copy) already beats it")

    print("\n--- oracle bound: perfect prediction on the k worst lane-B rows ---")
    nA, sseA = int((both.lane == "A").sum()), float((e[both.lane.values == "A"] ** 2).sum())
    base = float(((pb - y) ** 2).sum())
    order = np.argsort((pb - y) ** 2)[::-1]
    for k in (1, 2, 5, 9, 20):
        s = base - float(((pb[order[:k]] - y[order[:k]]) ** 2).sum())
        print(f"  k={k:2d}: lane B {np.sqrt(s/len(y)):7.1f}  composite {np.sqrt((sseA + s)/(nA + len(y))):8.4f}")


if __name__ == "__main__":
    main()
