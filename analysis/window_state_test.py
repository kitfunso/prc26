"""Label-free window state: the estimator docs/STATE_ORACLE.md asks for (its A4 oracle is the ceiling, +7.98 s).
features.window_state computes it; feat.parquet carries the 20 shipped columns (features.WZ), this recomputes all 40.
`PYTHONPATH=C:/Users/skf_s/prc26 python analysis/window_state_test.py [variant ...]` (default: all variants)"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from data import CACHE, rmse
from features import WS_SCALES as SCALES, build, window_state
from phys_tail import dayvar_features
from phys_tail_test import harness_check, run_variant
from state_oracle import oracle_columns
from train import FOLDS, HOLDOUT, boot_delta

STATS = ["n", "dz", "dz_sd", "hi", "lo", "dae", "dae_sd", "dae0", "dal", "dal_sd"]
WZ = [f"wz_{s}" for s in STATS[:8]]
WZ2 = ["wa_n", "wa_dz", "wa_dae", "wa_dae_sd", "wz30_dz", "wz30_dae", "wz30_dae_sd",
       "wz3h_dz", "wz3h_dae", "wz3h_dae_sd", "wz_dal", "wz_dal_sd"]
DISP = ["day_dae_std", "h_dae_std", "day_dal_std", "h_dal_std"]
VARIANTS = {"wz": WZ, "wz_disp": WZ + DISP, "wz2": WZ + WZ2}


def visibility(a: pd.DataFrame, cols: list[str]) -> None:
    """How much of the oracle window state the label-free stats can see: per-column corr and a linear R^2."""
    orc = oracle_columns(a)[0]
    X = np.column_stack([np.ones(len(a)), a[cols].values])
    for col in ("win_tail_shr", "win_physc_mean"):
        y = orc[col].values
        ok = np.isfinite(X).all(1) & np.isfinite(y)
        beta = np.linalg.lstsq(X[ok], y[ok], rcond=None)[0]
        r2 = 1 - np.var(y[ok] - X[ok] @ beta) / np.var(y[ok])
        corr = a[cols].corrwith(orc[col])
        print(f"{col} ({len(cols)} cols): linear R2 {r2:.3f} on {ok.sum()} rows; corr "
              + ", ".join(f"{k} {v:+.3f}" for k, v in corr.items()), flush=True)


def main(names: list[str]) -> None:
    f = build()
    ws = pd.concat([window_state(f, k, w, p) for p, (k, w) in SCALES.items()], axis=1)
    f = pd.concat([f.drop(columns=ws.columns, errors="ignore"), ws], axis=1)
    f = f[f["split"] == "train"]
    f = pd.concat([f, dayvar_features(f)], axis=1)
    a = f[f["has_plan"] == 1]
    for cols in {tuple(VARIANTS[n]) for n in names}:
        visibility(a, list(cols))
    ta, ha = a[~a["month"].isin(HOLDOUT)].copy(), a[a["month"].isin(HOLDOUT)].copy()
    harness_check(ta)
    base_oof = pd.read_parquet(CACHE / "ra_lgb_arr_oof.parquet").set_index("id")["oof"].reindex(ta["id"].values).values
    ref = {"ra_lgb_arr": "ra_lgb_arr_holdout.parquet", "full_disp": "analysis_phys_tail_full_disp_holdout.parquet",
           "wz": "analysis_window_state_wz_holdout.parquet"}
    ref = {k: CACHE / v for k, v in ref.items() if (CACHE / v).exists()}
    ref = {k: pd.read_parquet(v).set_index("id")["p"].reindex(ha["id"].values).values for k, v in ref.items()}
    y, day = ha["y"].values, ha["day"].values
    for name in names:
        oof, ph = run_variant(name, VARIANTS[name], ta, ha)
        pd.DataFrame({"id": ta["id"].values, "oof": oof}).to_parquet(CACHE / f"analysis_window_state_{name}_oof.parquet")
        pd.DataFrame({"id": ha["id"].values, "p": ph}).to_parquet(CACHE / f"analysis_window_state_{name}_holdout.parquet")
        better = sum(rmse(ta["y"].values[va], oof[va]) < rmse(ta["y"].values[va], base_oof[va])
                     for va in (ta["month"].isin(m).values for m in FOLDS))
        print(f"[{name}] holdout {rmse(y, ph):.3f}, folds better than ra_lgb_arr {better}/5", flush=True)
        for k, p0 in ref.items():
            if k != name:
                point, lo, hi = boot_delta(y, p0, ph, day)
                print(f"[{name}] vs {k} ({rmse(y, p0):.3f}): {point:+.3f} [{lo:+.3f}, {hi:+.3f}]", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or list(VARIANTS))
