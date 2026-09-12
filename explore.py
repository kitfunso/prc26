"""Steps 1-3 of PLAN.md: how the off-block was recorded, and what the test file itself constrains."""
from __future__ import annotations

import numpy as np
import pandas as pd

from data import PLAN_STAMPS, departures, load_raw, rmse, secs

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


def stamp_table(dep: pd.DataFrame) -> pd.DataFrame:
    """Per airport: how close each candidate stamp is to the recorded off-block (train rows only)."""
    rows = []
    y = dep["TAXITIME_SEC_mvt"]
    for apt, g in dep.groupby("ADEP_mvt"):
        r = {"apt": apt, "n": len(g), "y_mean": g["TAXITIME_SEC_mvt"].mean(), "y_p99": g["TAXITIME_SEC_mvt"].quantile(.99),
             "y>1h%": 100 * (g["TAXITIME_SEC_mvt"] > 3600).mean(), "y>20h%": 100 * (g["TAXITIME_SEC_mvt"] > 72000).mean()}
        for s in PLAN_STAMPS + ["SCHED_TIME_UTC_mvt"]:
            e = secs(g["BLOCK_TIME_UTC_mvt"], g[s])
            k = s.split("_")[0]
            r[f"{k}_cov%"] = 100 * e.notna().mean()
            r[f"{k}_copy%"] = 100 * (e.abs() < 60).mean()
            r[f"{k}_rmse"] = rmse(g["TAXITIME_SEC_mvt"][e.notna()], secs(g["MVT_TIME_UTC_mvt"], g[s])[e.notna()])
        rows.append(r)
    return pd.DataFrame(rows).set_index("apt").round(1)


def stand_bounds(df: pd.DataFrame) -> pd.DataFrame:
    """For each departure: in-block of the previous and next arrival at the same stand (both phases, all splits)."""
    arr = df[df["PHASE_mvt"] == "ARR"][["ADEP_mvt", "STAND_mvt", "BLOCK_TIME_UTC_mvt", "AIRCRAFT_TYPE_mvt"]].dropna(subset=["BLOCK_TIME_UTC_mvt"])
    arr = arr.rename(columns={"BLOCK_TIME_UTC_mvt": "arr_in", "AIRCRAFT_TYPE_mvt": "arr_type"}).sort_values("arr_in")
    dep = departures(df).sort_values("MVT_TIME_UTC_mvt")
    key = ["ADEP_mvt", "STAND_mvt"]
    prev = pd.merge_asof(dep[key + ["MVT_ID_mvt", "MVT_TIME_UTC_mvt"]], arr, left_on="MVT_TIME_UTC_mvt", right_on="arr_in",
                         by=key, direction="backward").rename(columns={"arr_in": "prev_in", "arr_type": "prev_type"})
    nxt = pd.merge_asof(dep[key + ["MVT_ID_mvt", "MVT_TIME_UTC_mvt"]], arr, left_on="MVT_TIME_UTC_mvt", right_on="arr_in",
                        by=key, direction="forward").rename(columns={"arr_in": "next_in", "arr_type": "next_type"})
    out = dep.merge(prev[["MVT_ID_mvt", "prev_in", "prev_type"]], on="MVT_ID_mvt").merge(nxt[["MVT_ID_mvt", "next_in", "next_type"]], on="MVT_ID_mvt")
    # An arrival that parked at our stand BEFORE we took off but AFTER we should have left bounds our off-block from above.
    # merge_asof backward gives the last arrival before takeoff; if it is not "us" (the turnaround), it is the upper bound.
    out["t_prev"] = secs(out["MVT_TIME_UTC_mvt"], out["prev_in"])
    out["t_next"] = secs(out["next_in"], out["MVT_TIME_UTC_mvt"])
    return out


def main() -> None:
    df = load_raw()
    dep = departures(df)
    tr = dep[dep["split"] == "train"]
    lines = ["# Exploration (steps 1-3)\n", f"train departures {len(tr):,}; rank departures {(dep['split']=='rank').sum():,}\n"]

    lines += ["\n## 1. Off-block mechanism per airport (train)\n", "cov = stamp present, copy = |block - stamp| < 60 s, rmse = takeoff - stamp as the prediction\n",
              stamp_table(tr).to_markdown(), "\n"]
    a, b = tr[tr["has_plan"]], tr[~tr["has_plan"]]
    lines += ["\n### lanes (train)\n", f"lane A {len(a):,} rmse(mean)={rmse(a['TAXITIME_SEC_mvt'], np.full(len(a), a['TAXITIME_SEC_mvt'].mean())):.1f}",
              f"\nlane B {len(b):,} rmse(mean)={rmse(b['TAXITIME_SEC_mvt'], np.full(len(b), b['TAXITIME_SEC_mvt'].mean())):.1f}\n"]
    lines += ["\n### lane B per airport (train)\n", b.groupby("ADEP_mvt")["TAXITIME_SEC_mvt"].agg(["size", "mean", "median", lambda s: s.quantile(.99), "max"]).round(0).to_markdown(), "\n"]
    bs = secs(b["MVT_TIME_UTC_mvt"], b["SCHED_TIME_UTC_mvt"])
    b = b.assign(sched_copy=(b["TAXITIME_SEC_mvt"] - bs).abs() < 60, dayshift=b["TAXITIME_SEC_mvt"] > 72000)
    lines += ["\nlane B schedule-copy share per airport (|y - (takeoff - sched)| < 60 s):\n", b.groupby("ADEP_mvt")[["sched_copy", "dayshift"]].mean().mul(100).round(2).to_markdown(), "\n"]
    sse = (b["TAXITIME_SEC_mvt"] - b["TAXITIME_SEC_mvt"].mean()) ** 2
    lines += [f"\nlane B SSE share of schedule-copy rows {100 * sse[b['sched_copy']].sum() / sse.sum():.1f}%, of y > 6h rows {100 * sse[b['TAXITIME_SEC_mvt'] > 21600].sum() / sse.sum():.1f}%\n"]

    lines += ["\n## 2. Stand bounds from arrivals (train departures)\n"]
    sb = stand_bounds(df)
    st = sb[sb["split"] == "train"]
    lines += [f"prev arrival at stand found {100 * st['prev_in'].notna().mean():.1f}%, next {100 * st['next_in'].notna().mean():.1f}%\n"]
    # Upper bound bites when the previous arrival's in-block is later than our off-block would normally be.
    ub = st[st["prev_in"].notna()]
    viol = secs(ub["BLOCK_TIME_UTC_mvt"], ub["prev_in"])
    lines += [f"block - prev_in: median {viol.median():.0f} s, share < 0 (we left before the 'previous' arrival parked) {100 * (viol < 0).mean():.2f}%\n"]
    lines += ["\nt_prev = takeoff - prev arrival in-block, by taxi band (train):\n"]
    st = st.assign(band=pd.cut(st["TAXITIME_SEC_mvt"], [-1, 600, 1200, 1800, 3600, 7200, 21600, 1e9]))
    lines += [st.groupby("band", observed=True).agg(n=("MVT_ID_mvt", "size"), t_prev_med=("t_prev", "median"), t_prev_p10=("t_prev", lambda s: s.quantile(.1)),
                                                    share_tprev_lt_y=("t_prev", lambda s: np.nan)).round(0).to_markdown(), "\n"]
    lt = st["t_prev"] < st["TAXITIME_SEC_mvt"]
    lines += [f"\nshare where takeoff - prev_in < y (bound violated => stand ids ambiguous or towed): {100 * lt.mean():.2f}% overall; by band:\n",
              st.assign(lt=lt).groupby("band", observed=True)["lt"].mean().mul(100).round(2).to_markdown(), "\n"]

    lines += ["\n## 3. Rank file vs train: lane B and stamp coverage\n"]
    rk = dep[dep["split"] == "rank"]
    lines += [f"rank rows {len(rk):,}, lane B {(~rk['has_plan']).sum():,} ({100 * (~rk['has_plan']).mean():.2f}%), train lane B share {100 * (~tr['has_plan']).mean():.2f}%\n",
              "\nlane B rank rows per airport and month:\n", rk[~rk["has_plan"]].groupby(["ADEP_mvt", "month"]).size().unstack(fill_value=0).to_markdown(), "\n",
              "\nlane B train rows per airport and month:\n", tr[~tr["has_plan"]].groupby(["ADEP_mvt", "month"]).size().unstack(fill_value=0).to_markdown(), "\n"]
    out = "\n".join(lines)
    (df_path := __import__("pathlib").Path("docs")).mkdir(exist_ok=True)
    (df_path / "EXPLORE.md").write_text(out, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
