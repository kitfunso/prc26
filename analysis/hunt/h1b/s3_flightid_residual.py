"""H1b item 2 (deep dive): after v34's OOF prediction, is there leftover signal in FLIGHT_ID_mvt?
If v34's residual (y - oof) still correlates with FLIGHT_ID's within-(apt,day) rank, FLIGHT_ID carries
information the model does not currently use. Dev months only (v34 OOF is CV over the 10 dev months).
Run: PYTHONPATH=. python analysis/hunt/h1b/s3_flightid_residual.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import DEV_MONTH_FILES, classify_offblock, read_month


def pooled_rank_corr(df: pd.DataFrame, key: list[str], a: str, b: str) -> float:
    g = df.groupby(key)
    ra = g[a].rank()
    rb = g[b].rank()
    return float(ra.corr(rb))


def main() -> None:
    oof = pd.concat([
        pd.read_parquet("data/cache/recal_oof_v34.parquet", columns=["id", "oof"]),
        pd.read_parquet("data/cache/laneb_oof.parquet", columns=["id", "oof"]),
    ], ignore_index=True)
    print("v34 OOF rows (lane A + lane B):", len(oof))

    parts = []
    for path in DEV_MONTH_FILES:
        df = read_month(path)
        df["cls"] = classify_offblock(df)
        df = df.rename(columns={"MVT_ID_mvt": "id", "FLIGHT_ID_mvt": "fid"})
        parts.append(df[["id", "fid", "apt", "day", "cls"]])
    d = pd.concat(parts, ignore_index=True)

    m = d.merge(oof, on="id", how="inner")
    y = pd.read_parquet("data/cache/feat.parquet", columns=["id", "y", "has_plan"])
    m = m.merge(y, on="id", how="inner")
    print("matched rows (raw dev x oof x feat):", len(m))
    m["resid"] = m["y"] - m["oof"]

    print("\n--- overall ---")
    print("within(apt,day) rank-corr(FLIGHT_ID, v34 residual):", pooled_rank_corr(m, ["apt", "day"], "fid", "resid"))
    print("within(apt,day) rank-corr(FLIGHT_ID, y)            :", pooled_rank_corr(m, ["apt", "day"], "fid", "y"))
    print("RMSE v34 OOF (dev, this matched set):", float(np.sqrt((m["resid"] ** 2).mean())))

    print("\n--- by off-block class ---")
    for cls, sub in m.groupby("cls"):
        if len(sub) < 200:
            print(f"{cls}: n={len(sub)} too few")
            continue
        c = pooled_rank_corr(sub, ["apt", "day"], "fid", "resid")
        print(f"{cls}: n={len(sub)} ({len(sub)/len(m):.4%}) rank-corr(FLIGHT_ID, resid)={c:.4f} "
              f"resid SSE share={float((sub['resid']**2).sum()/(m['resid']**2).sum()):.4%}")

    print("\n--- by lane (has_plan) ---")
    for hp, sub in m.groupby("has_plan"):
        c = pooled_rank_corr(sub, ["apt", "day"], "fid", "resid")
        print(f"has_plan={hp}: n={len(sub)} rank-corr(FLIGHT_ID, resid)={c:.4f} "
              f"resid SSE share={float((sub['resid']**2).sum()/(m['resid']**2).sum()):.4%}")

    print("\n--- large-residual subset (|resid| > 1800s) ---")
    big = m[m["resid"].abs() > 1800]
    print(f"n={len(big)} ({len(big)/len(m):.4%}) SSE share={float((big['resid']**2).sum()/(m['resid']**2).sum()):.4%}")
    print("rank-corr(FLIGHT_ID, resid) on this subset:", pooled_rank_corr(big, ["apt", "day"], "fid", "resid"))


if __name__ == "__main__":
    main()
