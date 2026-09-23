"""Label-free: does v34 predict far outside what it predicts on the 2025 holdout? Per airport, rank vs holdout tails."""
import numpy as np
import pandas as pd

C = "data/cache/"
f = pd.read_parquet(C + "feat.parquet", columns=["id", "split", "month", "has_plan", "y", "apt", "z_sched", "z_aobt", "z_eobt"])
hold = pd.concat([pd.read_parquet(C + "recal_holdout_v34.parquet"), pd.read_parquet(C + "laneb_holdout.parquet")])
rank = pd.read_parquet("data/merry-octopus_v34.parquet").rename(columns={"MVT_ID_mvt": "id", "TAXITIME_SEC_mvt": "p"})
h = f.merge(hold, on="id")
r = f.merge(rank, on="id")
print("holdout rows", len(h), "rank rows", len(r))
rows = []
for apt in sorted(f["apt"].astype(str).unique()):
    hp, rp, hy = h.loc[h.apt.astype(str) == apt, "p"], r.loc[r.apt.astype(str) == apt, "p"], h.loc[h.apt.astype(str) == apt, "y"]
    rows.append({"apt": apt, "h_n": len(hp), "r_n": len(rp), "h_p999": hp.quantile(.999), "r_p999": rp.quantile(.999),
                 "h_max": hp.max(), "r_max": rp.max(), "y_p999": hy.quantile(.999), "h_gt7200": int((hp > 7200).sum()),
                 "r_gt7200": int((rp > 7200).sum()), "y_gt7200": int((hy > 7200).sum())})
print(pd.DataFrame(rows).round(0).to_string(index=False))
lim = h.groupby(h.apt.astype(str))["p"].max()
r["h_max"] = r.apt.astype(str).map(lim)
out = r[r.p > r.h_max].sort_values("p", ascending=False)
print(f"\nrank rows predicted above the airport's holdout max: {len(out)}")
print(out[["id", "month", "apt", "has_plan", "z_aobt", "z_eobt", "z_sched", "p", "h_max"]].head(30).round(0).to_string(index=False))
