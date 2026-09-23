"""Counted holdout reads (months 1 and 7 of 2025): one per pre-registered candidate, refused if already logged.
B1: LIRF lane B hull rows with a garbled flight number and 4 h <= zs <= 12 h are schedule copies (bet_hull.py file).
L1: lane A rows whose whole flight plan (LOBT, IOBT, EOBT_1, AOBT_3) sits over 12 h before takeoff. Compares v34 with
p = MVT - AOBT_3 (day-sized) and with p = the airport's mean dev lane A taxi (normal).
Run: PYTHONPATH=. python analysis/hunt/h1a/read_holdout.py B1|L1   (appends to holdout_reads.log)"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from common import OUT, movements, v34_holdout
from train import boot_delta

LOG = OUT / "holdout_reads.log"
BIG = 43200
PLAN = ["LOBT_flt", "IOBT_flt", "EOBT_1_flt", "AOBT_3_flt"]


def frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = movements(mask_holdout=False)
    dep = df[df["PHASE_mvt"] == "DEP"]
    h = v34_holdout().merge(dep[dep["ho"]], left_on="id", right_on="MVT_ID_mvt", how="inner")
    h["day"] = (h["MVT_TIME_UTC_mvt"] // 86400).astype(int)
    h["zs"] = h["MVT_TIME_UTC_mvt"] - h["SCHED_TIME_UTC_mvt"]
    return h, dep[dep["dev"]]


def candidates(name: str, h: pd.DataFrame, dev: pd.DataFrame) -> dict[str, pd.Series]:
    if name == "B1":
        o = pd.read_parquet(OUT / "bet_hull_holdout_override.parquet").set_index("id")["p"]
        return {"B1 copy bet": h["id"].map(o)}
    far = pd.concat([(h["MVT_TIME_UTC_mvt"] - h[c]) > BIG for c in PLAN], axis=1).all(axis=1) & (h["lane"] == "A")
    taxi = dev[dev["has_plan"].astype(bool) & (dev["y"] <= 3600)].groupby("apt")["y"].mean()
    return {"L1 day-sized (MVT - AOBT_3)": (h["MVT_TIME_UTC_mvt"] - h["AOBT_3_flt"]).where(far),
            "L1 normal taxi": h["apt"].map(taxi).where(far)}


def main(name: str) -> None:
    done = LOG.read_text().splitlines() if LOG.exists() else []
    if any(f" {name}:" in line for line in done):
        sys.exit(f"{name} was already read; the rule allows one read per candidate")
    h, dev = frame()
    y, p0 = h["y"].values, h["p"].values
    lines = [f"read #{sum(l.startswith('read #') for l in done) + 1} {name}: holdout rows {len(h)}"]
    for label, alt in candidates(name, h, dev).items():
        m = alt.notna().values
        p1 = np.where(m, alt.values, p0)
        t = h[m]
        cls = np.select([(t["y"] - t["zs"]).abs() <= 10, t["y"] > BIG], ["copy", "day-sized"], "normal")
        d_sse = (((y - p0) ** 2).sum() - ((y - p1) ** 2).sum()) / 1e6
        pt, lo, hi = boot_delta(y, p0, p1, h["day"].values)
        lines.append(f"  {label}: touched {m.sum()} rows, classes {pd.Series(cls).value_counts().to_dict()},"
                     f" SSE saved vs v34 {d_sse:+.1f}M, boot_delta RMSE {pt:+.4f} s [{lo:+.4f}, {hi:+.4f}]")
        for (_, r), pc in zip(t.iterrows(), p1[m]):
            lines.append(f"    id {r['id']:.0f} {r['apt']} {r['FLIGHT_mvt']} zs {r['zs']:.0f} y {r['y']:.0f}"
                         f" v34 {r['p']:.0f} candidate {pc:.0f}")
    print("\n".join(lines))
    with LOG.open("a") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
