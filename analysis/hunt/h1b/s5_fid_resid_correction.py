"""H1b item 2, the deployable test: fit a correction to v34's OOF residual as a function of
FLIGHT_ID's within-(apt,day) percentile rank -- the only form transferable to rank rows, since a
rank row's neighbours' true y are never known (unlike a LOO neighbour-lookup, which needs labels
that don't exist at rank time). Fit on 8 dev months, evaluate held out on the other 2 dev months.
A parallel correction from HOUR-of-day rank (a feature v34 already sees) is the control: if FLIGHT_ID
does no better than a feature already in the model, there is no new signal, just GBM under-fit.
Run: PYTHONPATH=. python analysis/hunt/h1b/s5_fid_resid_correction.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import DEV_MONTH_FILES, read_month

FIT_MONTHS = [p for p in DEV_MONTH_FILES if any(t in p for t in
              ("2025-02", "2025-03", "2025-04", "2025-05", "2025-06", "2025-08", "2025-09", "2025-10"))]
TEST_MONTHS = [p for p in DEV_MONTH_FILES if p not in FIT_MONTHS]


def pct_rank(df: pd.DataFrame, col: str, key: list[str]) -> pd.Series:
    g = df.groupby(key)[col]
    return g.rank() / g.transform("count")


def fit_bins(fit_df: pd.DataFrame, rank_col: str, n_bins: int) -> pd.Series:
    b = pd.Series(pd.qcut(fit_df[rank_col], n_bins, labels=False, duplicates="drop"), name="bin")
    return fit_df["resid"].groupby(b).mean()  # index = bin id -> mean resid


def apply_bins(test_df: pd.DataFrame, rank_col: str, n_bins: int, lut: pd.Series) -> np.ndarray:
    edges = np.linspace(0, 1, n_bins + 1)
    b = np.clip(np.digitize(test_df[rank_col].to_numpy(), edges[1:-1]), 0, lut.index.max())
    return lut.reindex(b).to_numpy()


def load(paths: list[str]) -> pd.DataFrame:
    parts = []
    for path in paths:
        df = read_month(path)
        df = df.rename(columns={"MVT_ID_mvt": "id", "FLIGHT_ID_mvt": "fid"})
        parts.append(df[["id", "fid", "apt", "day"]])
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    oof = pd.concat([
        pd.read_parquet("data/cache/recal_oof_v34.parquet", columns=["id", "oof"]),
        pd.read_parquet("data/cache/laneb_oof.parquet", columns=["id", "oof"]),
    ], ignore_index=True)
    y = pd.read_parquet("data/cache/feat.parquet", columns=["id", "y", "hour"])

    fit_raw = load(FIT_MONTHS)
    test_raw = load(TEST_MONTHS)
    fit = fit_raw.merge(oof, on="id").merge(y, on="id")
    test = test_raw.merge(oof, on="id").merge(y, on="id")
    fit["resid"] = fit["y"] - fit["oof"]
    test["resid"] = test["y"] - test["oof"]

    fit["r_fid"] = pct_rank(fit, "fid", ["apt", "day"])
    fit["r_hour"] = pct_rank(fit, "hour", ["apt", "day"])
    test["r_fid"] = pct_rank(test, "fid", ["apt", "day"])
    test["r_hour"] = pct_rank(test, "hour", ["apt", "day"])

    base_sse = float((test["resid"] ** 2).sum())
    print(f"fit months={len(FIT_MONTHS)} test months={len(TEST_MONTHS)}")
    print(f"fit n={len(fit)} test n={len(test)}")
    print(f"test baseline (v34 OOF) SSE={base_sse:.6g} RMSE={np.sqrt(base_sse/len(test)):.4f}")

    for n_bins in (10, 20, 40):
        for rank_col in ("r_fid", "r_hour"):
            table = fit_bins(fit, rank_col, n_bins)
            corr = apply_bins(test, rank_col, n_bins, table)
            new_resid = test["resid"].to_numpy() - corr
            sse = float((new_resid ** 2).sum())
            print(f"bins={n_bins:3d} corrector={rank_col:6s} new SSE={sse:.6g} "
                  f"delta={base_sse - sse:+.6g} ({(base_sse-sse)/1e6:+.2f}M)")


if __name__ == "__main__":
    main()
