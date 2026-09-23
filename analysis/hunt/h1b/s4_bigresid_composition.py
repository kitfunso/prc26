"""H1b: what IS the large-residual (|resid|>1800s) dev subset? Composition by y pattern,
and does a FLIGHT_ID neighbor lookup (leave-one-out, same apt+day) predict these rows' y
better than v34's OOF? Dev months only.
Run: PYTHONPATH=. python analysis/hunt/h1b/s4_bigresid_composition.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import DEV_MONTH_FILES, read_month


def main() -> None:
    oof = pd.concat([
        pd.read_parquet("data/cache/recal_oof_v34.parquet", columns=["id", "oof"]),
        pd.read_parquet("data/cache/laneb_oof.parquet", columns=["id", "oof"]),
    ], ignore_index=True)

    parts = []
    for path in DEV_MONTH_FILES:
        df = read_month(path)
        df = df.rename(columns={"MVT_ID_mvt": "id", "FLIGHT_ID_mvt": "fid"})
        parts.append(df[["id", "fid", "apt", "day"]])
    d = pd.concat(parts, ignore_index=True)

    y = pd.read_parquet("data/cache/feat.parquet", columns=["id", "y", "has_plan"])
    m = d.merge(oof, on="id", how="inner").merge(y, on="id", how="inner")
    m["resid"] = m["y"] - m["oof"]

    big = m[m["resid"].abs() > 1800].copy()
    print(f"large-residual subset n={len(big)}")
    near_day = (big["y"].abs() - 86400).abs() < 900
    print(f"  |y| within 900s of one day (86400s): {near_day.sum()} ({near_day.mean():.2%})")
    near_2day = (big["y"].abs() - 172800).abs() < 900
    print(f"  |y| within 900s of two days (172800s): {near_2day.sum()} ({near_2day.mean():.2%})")
    print(f"  y stats: min={big['y'].min():.0f} p1={big['y'].quantile(.01):.0f} "
          f"median={big['y'].median():.0f} p99={big['y'].quantile(.99):.0f} max={big['y'].max():.0f}")
    print(f"  oof stats: min={big['oof'].min():.0f} median={big['oof'].median():.0f} max={big['oof'].max():.0f}")
    print(f"  has_plan share: {big['has_plan'].mean():.2%} (all-dev has_plan share: {m['has_plan'].mean():.2%})")

    # FLIGHT_ID leave-one-out neighbor mean of y, within (apt, day), centered window of 2W+1 by FLIGHT_ID rank.
    # Vectorized exact leave-one-out mean: (rolling sum - self) / (rolling count - 1).
    W = 3
    m = m.sort_values(["apt", "day", "fid"]).reset_index(drop=True)
    roll = m.groupby(["apt", "day"])["y"].rolling(window=2 * W + 1, center=True, min_periods=2).agg(["sum", "count"])
    roll = roll.reset_index(level=[0, 1], drop=True)
    m["fid_nn_pred"] = (roll["sum"] - m["y"]) / (roll["count"] - 1)
    ok = m.dropna(subset=["fid_nn_pred"])
    print(f"\nFLIGHT_ID neighbor (+-{W}) median-of-y predictor, n={len(ok)}")
    print("  RMSE v34 OOF          :", float(np.sqrt(((ok['y'] - ok['oof']) ** 2).mean())))
    print("  RMSE FLIGHT_ID nbr    :", float(np.sqrt(((ok['y'] - ok['fid_nn_pred']) ** 2).mean())))
    okbig = ok[(ok["y"] - ok["oof"]).abs() > 1800]
    print(f"  on the large-v34-residual subset (n={len(okbig)}):")
    print("    RMSE v34 OOF        :", float(np.sqrt(((okbig['y'] - okbig['oof']) ** 2).mean())))
    print("    RMSE FLIGHT_ID nbr  :", float(np.sqrt(((okbig['y'] - okbig['fid_nn_pred']) ** 2).mean())))
    ok.to_parquet("analysis/hunt/h1b/_tmp_fidnn_dev.parquet")


if __name__ == "__main__":
    main()
