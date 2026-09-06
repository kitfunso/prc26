"""Day-block bootstrap of the RMSE noise bar on the Jan+Jul 2025 lane-A holdout (protocol section 0)."""
import argparse

import duckdb
import numpy as np

HOLDOUT = """
SELECT MVT_TIME_UTC_mvt::date d, ADEP_mvt apt,
       epoch(MVT_TIME_UTC_mvt)-epoch(AOBT_3_flt)-TAXITIME_SEC_mvt AS err
FROM read_parquet('data/training_*.parquet')
WHERE PHASE_mvt='DEP' AND TAXITIME_SEC_mvt IS NOT NULL AND AOBT_3_flt IS NOT NULL
  AND (MVT_TIME_UTC_mvt < '2025-02-01'
       OR (MVT_TIME_UTC_mvt >= '2025-07-01' AND MVT_TIME_UTC_mvt < '2025-08-01'))"""


def rmse(e: np.ndarray) -> float:
    return float(np.sqrt(np.mean(e ** 2)))


def main(reps: int, block: int, scored: int) -> None:
    df = duckdb.sql(HOLDOUT).df()
    err = df.err.values.astype(float)
    err_b = err - df.groupby("apt").err.transform("mean").values
    print("holdout rows:", len(df), "| days:", df.d.nunique(), "| day-airport blocks:", df.groupby(["d", "apt"]).ngroups)
    print("raw AOBT_3 residual RMSE:", round(rmse(err), 1), "| per-airport-debiased:", round(rmse(err_b), 1))
    rng = np.random.default_rng(0)
    ud, inv = np.unique(df.d.values, return_inverse=True)
    grp = inv // block
    nb = int(grp.max()) + 1
    by_block = [np.flatnonzero(grp == b) for b in range(nb)]
    levels, deltas = [], []
    for _ in range(reps):
        ix = np.concatenate([by_block[b] for b in rng.integers(0, nb, nb)])
        if scored and len(ix) > scored:
            ix = rng.choice(ix, scored, replace=False)
        levels.append(rmse(err[ix]))
        deltas.append(rmse(err[ix]) - rmse(err_b[ix]))
    print(f"block={block} day(s), resample to {scored or 'none'}: "
          f"SE(level)={np.std(levels):.1f} s, band {np.percentile(levels, 2.5):.1f}-{np.percentile(levels, 97.5):.1f}")
    print(f"paired SE(delta)={np.std(deltas):.2f} s (delta {np.mean(deltas):.2f})")
    if scored:
        iid = [rmse(err[rng.choice(len(err), scored, replace=False)]) for _ in range(reps)]
        print(f"iid-row SE(level)={np.std(iid):.1f} s | gaussian formula={rmse(err) / np.sqrt(2 * scored):.2f} s")
    tail = np.abs(err) > 600
    print(f"tail rows {100 * tail.mean():.1f}% carry {100 * np.sum(err[tail] ** 2) / np.sum(err ** 2):.1f}% of SSE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--block", type=int, default=1, help="consecutive days per bootstrap block")
    ap.add_argument("--scored", type=int, default=0, help="subsample each draw to this many rows; 0 (the default) keeps every row, because the board scores all 344,841 since 2026-09-04")
    a = ap.parse_args()
    main(a.reps, a.block, a.scored)
