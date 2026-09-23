"""Rows whose schedule gap z_sched is over 70,000 s: split, lane, label and what v34 predicts on them."""
import pandas as pd

C = "data/cache/"
f = pd.read_parquet(C + "feat.parquet", columns=["id", "split", "month", "has_plan", "y", "apt", "z_sched", "z_aobt", "z_eobt"])
print(f["split"].value_counts().to_dict())
big = f[f["z_sched"] > 70_000].copy()
print("rows with z_sched > 70,000 by split, airport, has_plan:")
print(big.groupby(["split", "apt", "has_plan"], observed=True).size().to_string())

preds = []
for path, col in [(C + "recal_oof_v34.parquet", None), (C + "laneb_oof.parquet", None),
                  (C + "recal_holdout_v34.parquet", None), (C + "laneb_holdout.parquet", None),
                  ("data/merry-octopus_v34.parquet", None)]:
    p = pd.read_parquet(path).rename(columns={"MVT_ID_mvt": "id"})
    val = [c for c in p.columns if c != "id"][0]
    preds.append(p[["id", val]].rename(columns={val: "p_" + path.split("/")[-1].split(".")[0]}))
for p in preds:
    big = big.merge(p, on="id", how="left")
pd.set_option("display.width", 250)
print(big.sort_values(["split", "apt", "z_sched"]).to_string(index=False))
