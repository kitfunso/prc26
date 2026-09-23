"""H1b item 5: time zones / DST. Does off-block ever look like a stamp copy that's off by exactly
+-3600s (a one-hour local/UTC mixup classify_offblock's tol=60s would miss), and does this cluster
on the 2025 DST transition dates (30 Mar spring-forward, 26 Oct fall-back) or at LTFM (Istanbul,
UTC+3 year-round, the one monitored airport with no DST)? Train dev rows only (BLOCK_TIME known).
Run: PYTHONPATH=. python analysis/hunt/h1b/s8_timezones.py
"""
from __future__ import annotations

import pandas as pd

from common import DEV_MONTH_FILES, read_month
from data import secs

DST_DATES = {"2025-03-30", "2025-10-26"}
STAMPS = {"AOBT": "AOBT_3_flt", "EOBT": "EOBT_1_flt", "SCHED": "SCHED_TIME_UTC_mvt"}


def main() -> None:
    parts = [read_month(p) for p in DEV_MONTH_FILES]
    d = pd.concat(parts, ignore_index=True)
    y = secs(d["MVT_TIME_UTC_mvt"], d["BLOCK_TIME_UTC_mvt"])

    print(f"dev DEP rows: {len(d)}")
    print("\n--- y mod 3600 (whole-hour clustering check) ---")
    mod = y % 3600
    near_edge = (mod < 60) | (mod > 3540)
    print(f"y within 60s of a whole-hour boundary: {near_edge.mean():.4%} (uniform-random expectation: 3.33%)")

    print("\n--- near-hour stamp-copy check (gap to AOBT/EOBT/SCHED within 60s of +-3600s) ---")
    hour_hit = pd.Series(False, index=d.index)
    for name, col in STAMPS.items():
        gap = secs(d["BLOCK_TIME_UTC_mvt"], d[col])
        hit = ((gap - 3600).abs() <= 60) | ((gap + 3600).abs() <= 60)
        print(f"{name}: BLOCK_TIME within 60s of {name} +-3600s: {hit.sum()} ({hit.mean():.4%})")
        hour_hit |= hit
    d["hour_shift"] = hour_hit
    print(f"any stamp, union: {hour_hit.sum()} ({hour_hit.mean():.4%})")

    print("\n--- hour_shift rate by day, DST transition dates vs rest ---")
    d["is_dst_date"] = d["day"].isin(DST_DATES)
    by_dst = d.groupby("is_dst_date")["hour_shift"].agg(["mean", "size"])
    print(by_dst)

    print("\n--- hour_shift rate by airport, LTFM (no DST) vs the other 9 (DST-observing) ---")
    d["is_ltfm"] = d["apt"] == "LTFM"
    by_apt = d.groupby("is_ltfm")["hour_shift"].agg(["mean", "size"])
    print(by_apt)

    print("\n--- hour_shift rate, LTFM x DST-date interaction ---")
    print(d.groupby(["is_ltfm", "is_dst_date"])["hour_shift"].agg(["mean", "size"]))

    print("\n--- the +-3 days around each DST date, all airports (any localized spike?) ---")
    for dst_day in sorted(DST_DATES):
        window = pd.date_range(pd.Timestamp(dst_day) - pd.Timedelta(days=3),
                                pd.Timestamp(dst_day) + pd.Timedelta(days=3)).astype(str)
        sub = d[d["day"].isin(window)]
        print(f"\n{dst_day} +-3d:")
        print(sub.groupby("day")["hour_shift"].agg(["mean", "size"]))


if __name__ == "__main__":
    main()
