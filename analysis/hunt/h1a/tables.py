"""Every report table that no other script prints: A rows with zs > 12 h by class; B v34's dev residual by decile of
the congestion features (needs arrcong.py); C the LIRF lane B cell hedge by zs band and flight-number shape (needs
copymodel.py); D lane A rows whose flight-plan AOBT sits over 1 h before takeoff, plus the ITY340 what-if; E the ten
dev rows with the largest v34 squared error; F LIRF lane B rows that cross UTC midnight with zs < 12 h.
Run: PYTHONPATH=. python analysis/hunt/h1a/tables.py"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import OUT, movements, v34_dev, v34_rank
from copymodel import FOLDS

BIG = 43200
ITY340 = 213147416


def part_a(d: pd.DataFrame) -> None:
    b = d[d["zs"] > BIG].copy()
    b["cls"] = np.select([b["y"].isna(), (b["y"] - b["zs"]).abs() <= 10, b["y"] > BIG], ["?", "copy", "day-sized"],
                         "normal")
    print("A. rows with zs > 12 h: label class by part, airport, lane B flag\n",
          b.groupby(["part", "apt", "laneB", "cls"]).size().unstack(fill_value=0).to_string())
    b["lirf_b"] = (b["apt"] == "LIRF") & (b["laneB"] == 1)
    print(" mean v34 p / zs on those rows, by part and LIRF lane B flag\n",
          (b["p"] / b["zs"]).groupby([b["part"], b["lirf_b"]]).mean().round(3).unstack().to_string())


def part_b(d: pd.DataFrame) -> None:
    v = d[(d["part"] == "dev") & (d["y"] <= BIG)].copy()
    v["r"] = v["y"] - v["p"]
    for c in ("am60", "as60", "ds60", "dm60"):
        v["dec"] = v.groupby("apt")[c].transform(lambda s: pd.qcut(s.rank(method="first"), 10, labels=False))
        print(f"\nB. {c}: mean v34 residual (s) by within-airport decile\n",
              v.groupby(["apt", "dec"])["r"].mean().unstack().round(0).to_string())
        top = v[v["dec"] == 9].groupby("apt")["r"]
        gain = (top.apply(lambda r: (r ** 2).sum()) - top.apply(lambda r: ((r - r.mean()) ** 2).sum())) / 1e6
        print(f"   top-decile SSE removable by a mean shift, in-sample (M): {gain.round(1).to_dict()}")


def part_c() -> None:
    r = pd.read_parquet(OUT / "copyprob_lirf.parquet")
    b = r[r["lane"] == "B"].copy()
    b["shape"] = np.where(b["fmt"] == "plain", "plain", "garbled")
    b["zg"] = pd.cut(b["zs"] / 3600, [0.5, 1, 2, 3, 4, 12, 99]).astype(str)
    dv = b[b["part"] == "dev"].copy()
    taxi = float(dv[(dv["copy"] == 0) & (dv["y"] <= 3600)]["y"].mean())
    for f in FOLDS:
        m = dv["month"].isin(f)
        g = dv[~m].groupby(["zg", "shape"])["copy"].agg(["sum", "count"])
        k = pd.MultiIndex.from_frame(dv.loc[m, ["zg", "shape"]])
        q = (g["sum"].reindex(k).fillna(0).values + 1) / (g["count"].reindex(k).fillna(0).values + 2)
        dv.loc[m, "pH"] = q * dv.loc[m, "zs"] + (1 - q) * taxi
    t = dv.groupby(["zg", "shape"]).agg(n=("y", "size"), copy_rate=("copy", "mean"))
    t["sse_v34_M"] = dv.groupby(["zg", "shape"]).apply(lambda s: ((s["y"] - s["p"]) ** 2).sum() / 1e6, include_groups=False)
    t["sse_cell_M"] = dv.groupby(["zg", "shape"]).apply(lambda s: ((s["y"] - s["pH"]) ** 2).sum() / 1e6, include_groups=False)
    t["saved_M"] = t["sse_v34_M"] - t["sse_cell_M"]
    print("\nC. LIRF lane B cell hedge (Beta(1,1) copy rate of the other folds' cell), dev OOF\n", t.round(3).to_string())
    print(" holdout and rank rows per cell\n",
          b[b["part"] != "dev"].groupby(["part", "zg", "shape"]).size().unstack([0, 2], fill_value=0).to_string())


def part_d(df: pd.DataFrame) -> None:
    dep = df[(df["PHASE_mvt"] == "DEP") & df["has_plan"].astype(bool)].copy()
    dep["gA"] = dep["MVT_TIME_UTC_mvt"] - dep["AOBT_3_flt"]
    d = dep[dep["dev"] & (dep["gA"] > 3600)].copy()
    d["cls"] = np.select([(d["y"] - d["gA"]).abs() <= 60, d["y"] > 3600], ["y = MVT - AOBT_3", "other > 1 h"], "<= 1 h")
    d["gb"] = pd.cut(d["gA"] / 3600, [1, 2, 4, 12, 48, 1e6])
    print("\nD. dev lane A rows whose AOBT_3 is over 1 h before takeoff: class of y\n",
          d.groupby(["gb", "cls"], observed=True).size().unstack(fill_value=0).to_string(), "\n",
          d.groupby(["apt", "cls"]).size().unstack(fill_value=0).to_string())
    taxi = dep[dep["dev"] & (dep["apt"] == "LIRF") & (dep["y"] <= 3600)]["y"].mean()
    row = dep[dep["MVT_ID_mvt"] == ITY340].iloc[0]
    p = float(v34_rank().set_index("id").loc[ITY340, "p"])
    t = row["MVT_TIME_UTC_mvt"]
    truth = {"SCHED copy": t - row["SCHED_TIME_UTC_mvt"], "LOBT": t - row["LOBT_flt"], "EOBT_1": t - row["EOBT_1_flt"],
             "AOBT_3": t - row["AOBT_3_flt"], "normal taxi": taxi}
    print(f" ITY340 (rank {ITY340}): v34 {p:.0f}; SSE change (M) of p = zs against v34 if the label is ...")
    for k, yv in truth.items():
        print(f"   {k} y={yv:.0f}: v34 SE {(yv - p) ** 2 / 1e6:.1f}M, p=zs SE {(yv - truth['SCHED copy']) ** 2 / 1e6:.1f}M,"
              f" saved {((yv - p) ** 2 - (yv - truth['SCHED copy']) ** 2) / 1e6:+.1f}M")


def part_e(df: pd.DataFrame) -> None:
    d = df[(df["PHASE_mvt"] == "DEP") & df["dev"]].merge(v34_dev(), left_on="MVT_ID_mvt", right_on="id")
    d["se_B"] = (d["y"] - d["p"]) ** 2 / 1e9
    d["zs"] = d["MVT_TIME_UTC_mvt"] - d["SCHED_TIME_UTC_mvt"]
    print("\nE. ten dev rows with the largest v34 squared error\n",
          d.nlargest(10, "se_B")[["MVT_ID_mvt", "apt", "lane", "FLIGHT_mvt", "zs", "y", "p", "se_B"]].round(3).to_string())


def part_f(d: pd.DataFrame) -> None:
    b = d[(d["apt"] == "LIRF") & (d["laneB"] == 1) & (d["zs"] < BIG)].copy()
    b["dutc"] = (b["MVT_TIME_UTC_mvt"] - 900) // 86400 - b["SCHED_TIME_UTC_mvt"] // 86400
    b = b[b["dutc"] >= 1]
    b["cls"] = np.select([b["y"].isna(), (b["y"] - b["zs"]).abs() <= 10, b["y"] > BIG], ["?", "copy", "shifted"], "normal")
    print("\nF. LIRF lane B rows whose takeoff - 15 min falls on a later UTC date than the schedule (zs < 12 h)\n",
          b.groupby(["part", "cls"]).size().unstack(fill_value=0).to_string(),
          f"\n mean v34 p / zs by part: {(b['p'] / b['zs']).groupby(b['part']).mean().round(3).to_dict()}")


def main() -> None:
    pd.set_option("display.width", 250)
    d = pd.read_parquet(OUT / "arrcong_feat.parquet")
    part_a(d)
    part_b(d)
    part_c()
    part_f(d)
    df = movements()
    part_d(df)
    part_e(df)


if __name__ == "__main__":
    main()
