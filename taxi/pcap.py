"""PCAP: an upper cap on the PREDICTION, out of fold, both lanes (diagnostic, never a verdict).

C1 measured the floor at 0 s and promoted it; the symmetric question was never asked, and on a metric this dominated by a few extreme
rows it is where a cheap knob would pay if one existed. The CAP sweep in COVERAGE section 5 is a different thing: training-truth admission.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from taxi.data import TRAIN_MONTHS, load
from taxi.l2 import floored
from taxi.lbcal import oof as oof_b
from taxi.verdict import rmse

CAPS_A = (3000, 5000, 8000, 15000, 30000)
CAPS_B = (2000, 3000, 5000, 8000, 12000, 20000, 30000, 45000)


def oof_a() -> tuple[np.ndarray, np.ndarray]:
    o = pd.read_parquet("data/cache/mc2_oof.parquet")
    tr = load(TRAIN_MONTHS)
    m = tr[tr.has_fpl].reset_index(drop=True).merge(o[["id", "pred"]], on="id", how="inner")
    return floored(m.pred.values), m.y.values


def table(p: np.ndarray, y: np.ndarray, caps: tuple[int, ...]) -> list[str]:
    r0 = rmse(p - y)
    rows = [f"| none | 0 | {r0:.2f} | +0.00 |"]
    for c in caps:
        q = np.minimum(p, c)
        rows.append(f"| {c:,} | {int((p > c).sum()):,} | {rmse(q - y):.2f} | {r0 - rmse(q - y):+.2f} |")
    return rows


def main() -> None:
    pa, ya = oof_a()
    ob = oof_b()
    pb, yb = ob.pred.values, ob.y.values
    head = ["| cap (s) | rows capped | OOF RMSE (s) | gain (s) |", "|---|---|---|---|"]
    out = ["# PCAP: an upper cap on the prediction, out of fold (diagnostic, never a verdict)", "",
           "Regenerate with `python -m taxi.pcap`. No cell is declared and nothing here can decide a verdict. C1 measured the FLOOR at 0 s "
           "and promoted it; nothing had ever measured the cap, and `docs/LEV.md` section 3 shows the composite is carried by a handful of "
           "extreme rows, so a one-parameter clip is exactly the shape of knob that would pay there if one paid. Lane A is MC2's cached "
           "out-of-fold prediction on the ten training months, floored; lane B is the champion L8c refit per fold by `taxi.lbcal.oof`. "
           "Choosing on out-of-fold rows and not on the holdout is the point: a cap read off the verdict set would be free by construction.", "",
           f"## Lane A ({len(pa):,} rows)", "",
           f"Prediction max {pa.max():,.0f} s, p99.9 {np.percentile(pa, 99.9):,.0f}; truth max {ya.max():,.0f} s, "
           f"p99.9 {np.percentile(ya, 99.9):,.0f}.", "", *head, *table(pa, ya, CAPS_A), "",
           f"## Lane B ({len(pb):,} rows)", "",
           f"Prediction max {pb.max():,.0f} s, p99.9 {np.percentile(pb, 99.9):,.0f}; truth max {yb.max():,.0f} s, "
           f"p99.9 {np.percentile(yb, 99.9):,.0f}.", "", *head, *table(pb, yb, CAPS_B), "",
           "## Reading", "",
           "The cap is harmful at every level tried, on both lanes, and monotonically: the tighter the cap the worse the lane. It is not a "
           "knob with a bad setting, it is a knob with no live side. The reason is in the two summary lines above. Lane B's truth reaches "
           f"{yb.max():,.0f} s and its p99.9 is {np.percentile(yb, 99.9):,.0f} s against a predicted p99.9 of "
           f"{np.percentile(pb, 99.9):,.0f}, so the model's extreme predictions are landing on rows whose answers really are that large; "
           "clipping them turns a row that was roughly right into a row that is wrong by tens of thousands of seconds. Lane A never "
           "predicts above 15,000 s at all, so a cap there is either inert or destructive with nothing in between.", "",
           "What this closes: the clip family, both directions. The floor is measured and kept (C1), the cap is measured and dead, and "
           "neither needs asking again. What it does not license: any claim about the extreme rows themselves, which `docs/DE.md` and flag "
           "F21 price at zero by their own separate argument."]
    text = "\n".join(out) + "\n"
    with open("docs/PCAP.md", "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
