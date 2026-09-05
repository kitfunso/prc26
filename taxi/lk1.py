"""LK1 (S3 critique finding 6): anchor source counts, anchors after takeoff and the largest non-gap correlation with y, per month and lane, over 2025
and the ranking file; `tests/test_leak.py` asserts the same rules on February."""
from __future__ import annotations

import numpy as np
import pandas as pd

from taxi.data import EXTRA, FEATURES, PLAN, STAND, features, load
from taxi.s2 import RANKING

# The takeoff-gap family and the stand's on-block gap are the label's own scale (measured on Feb 2025); everything else stays under LIMIT.
GAPS = {"z_aobt", "z_eobt", "z_lobt", "z_sched", "z_anchor", "t_occ"}
LIMIT, MIN_ROWS = 0.5, 30


def correlations(d: pd.DataFrame) -> dict[str, float]:
    out = {}
    for c in (*FEATURES, *PLAN, *STAND, *EXTRA):
        if c in GAPS or c == "has_fpl" or not pd.api.types.is_numeric_dtype(d[c]):
            continue
        x = d[c].astype(float)
        ok = x.notna()
        if ok.sum() >= MIN_ROWS and x[ok].std() > 0:
            out[c] = abs(float(np.corrcoef(x[ok], d.y[ok])[0, 1]))
    return out


def sources(df: pd.DataFrame) -> dict[str, int]:
    has = df.has_fpl.values
    aobt = has & (df.z_aobt.values >= 0)
    eobt = has & ~aobt & (df.z_eobt.values >= 0)
    sched = ~aobt & ~eobt & (df.z_sched.values >= 0)
    return dict(rows=len(df), aobt=int(aobt.sum()), eobt=int(eobt.sum()), sched=int(sched.sum()), takeoff=int((~aobt & ~eobt & ~sched).sum()),
                after_takeoff=int((df.z_anchor.values < 0).sum()))


def main() -> None:
    lines = ["| month | rows | anchor = AOBT | EOBT | schedule | takeoff itself | anchors after takeoff | max non-gap corr, lane A | lane B |",
             "|---|---|---|---|---|---|---|---|---|"]
    worst, after = 0.0, 0
    for m in range(1, 13):
        df = load((m,))
        s = sources(df)
        top = []
        for lane in (True, False):
            r = correlations(df[df.has_fpl == lane])
            c = max(r, key=r.get)
            top.append(f"{c} {r[c]:.3f}")
            worst = max(worst, r[c])
        after += s["after_takeoff"]
        lines.append(f"| {m:02d} | {s['rows']:,} | {s['aobt']:,} | {s['eobt']:,} | {s['sched']:,} | {s['takeoff']:,} | {s['after_takeoff']} | {top[0]} | {top[1]} |")
        print(lines[-1], flush=True)
    s = sources(features([RANKING]))
    after += s["after_takeoff"]
    lines.append(f"| ranking | {s['rows']:,} | {s['aobt']:,} | {s['eobt']:,} | {s['sched']:,} | {s['takeoff']:,} | {s['after_takeoff']} | no label | no label |")
    text = "\n".join(["# LK1: anchor and label-leak check over 2025 and the ranking file", "",
                      "Regenerate with `python -m taxi.lk1`. Anchor rule after A1: the first non-negative of the AOBT, EOBT and schedule gaps, else the "
                      "takeoff itself. The correlation columns are the largest |corr(feature, y)| among the features outside the declared gap family "
                      f"({sorted(GAPS)}, limit {LIMIT}); `tests/test_leak.py` asserts the same on February.", "", *lines, "",
                      f"Largest non-gap correlation over the year: {worst:.3f} (limit {LIMIT}). Anchors after takeoff over the year and the ranking file: {after}."]) + "\n"
    with open("docs/LK1.md", "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
