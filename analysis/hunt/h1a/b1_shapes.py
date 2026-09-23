"""Session check on bet B1: dev copy rate by zs band and flight-number shape, 2026 counts, and the override file.
Run: PYTHONPATH=. python analysis/hunt/h1a/b1_shapes.py  (needs copyprob_lirf.parquet from copymodel.py)"""
import pandas as pd

c = pd.read_parquet("analysis/hunt/h1a/copyprob_lirf.parquet")
b = c[c["lane"] == "B"].copy()
b["band"] = pd.cut(b["zs"] / 3600, [0.5, 2, 4, 12, 24, 99])
dev = b[b["part"] == "dev"]
print("dev lane B LIRF DEP: copies / rows by zs band (h) and shape")
t = dev.groupby(["band", "fmt"], observed=True)["copy"].agg(["sum", "count"]).astype(int)
t["v34_p_over_zs"] = dev.groupby(["band", "fmt"], observed=True).apply(
    lambda s: (s["p"] / s["zs"]).median(), include_groups=False).round(3)
print(t.to_string())
print("\nrows by part in the 4-12 h band, by shape (label-free):")
print(b[b["band"].astype(str) == "(4.0, 12.0]"].groupby(["part", "fmt"]).size().unstack(fill_value=0).to_string())
o = pd.read_parquet("analysis/hunt/h1a/bet_hull_rank_override.parquet")
v = pd.read_parquet("data/merry-octopus_v34.parquet")
print("\noverride", o.dtypes.to_dict(), len(o), "| v34", v.dtypes.to_dict(), len(v))
print("override ids in v34:", int(o["MVT_ID_mvt"].isin(v["MVT_ID_mvt"]).sum()), "of", len(o))
