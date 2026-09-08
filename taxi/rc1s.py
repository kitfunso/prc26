"""RC1S: the ranking file with lane A blended SB3 + RC1 at the out-of-fold weight, lane B taken from the parent unchanged.

SB3's lane-A values are read from the parent file rather than refit, because GPU CatBoost fits do not reproduce (S3, S4), so the
blend is the only new arithmetic in the file. Build only; upload is a Keith gate.
"""
from __future__ import annotations

import hashlib
import sys

import numpy as np
import optuna
import pandas as pd

from taxi.champ import lane_a
from taxi.data import HOLDOUT_MONTHS, TRAIN_MONTHS, features, load
from taxi.l2 import floored
from taxi.nn1 import base_oof, blend_weight
from taxi.rc1 import STORE, fit_predict, oof, suggest
from taxi.s2 import RANKING
from taxi.submit import build_submission
from taxi.weather import join, load_metar

PARENT = "data/merry-octopus_v16.parquet"


def best_cell() -> dict:
    return suggest(optuna.trial.FixedTrial(optuna.load_study(study_name="rc1", storage=STORE).best_params))


def weight(cell: dict) -> float:
    """The out-of-fold weight `taxi.rc1` reports, recomputed here so the file reproduces from this module alone."""
    tra = lane_a()[0]
    p, base = oof(tra, cell), base_oof(tra)
    ok = np.isfinite(p) & np.isfinite(base)
    return blend_weight(p[ok], base[ok], tra.y.values[ok])


def main(out: str, parent: str = PARENT) -> None:
    cell, metar = best_cell(), load_metar()
    w = weight(cell)
    tr, rk = (join(d, metar) for d in (load(TRAIN_MONTHS + HOLDOUT_MONTHS), features([RANKING])))
    m = rk.has_fpl.values
    rc = fit_predict(tr[tr.has_fpl].reset_index(drop=True), rk[m].reset_index(drop=True), cell)
    pp = pd.read_parquet(parent)
    par = pd.Series(pp.TAXITIME_SEC_mvt.values.astype("float64"), index=pp.MVT_ID_mvt.values).reindex(rk.id.values)
    if par.isna().any():
        raise ValueError(f"{int(par.isna().sum())} ranking ids missing from {parent}")
    vals, sb3 = par.to_numpy(float).copy(), par.to_numpy(float)[m]
    vals[m] = floored(w * rc + (1 - w) * sb3)
    sub = build_submission(pd.Series(vals, index=rk.id.values), out, floor=0.0)
    shown = {k: (v if k != "cols" else len(v)) for k, v in cell.items()}
    print(f"lane A: {int(m.sum())} rows, w {w:.6f}, RC1 mean {rc.mean():.1f}, SB3 mean {sb3.mean():.1f}, "
          f"blend mean {vals[m].mean():.1f}, rows moved by 1 s or more {int((np.abs(vals[m] - sb3) >= 1).sum())}, "
          f"largest move {np.abs(vals[m] - sb3).max():.1f} s", flush=True)
    print(f"lane B: {int((~m).sum())} rows from {parent}, unchanged")
    print(f"{out}: {len(sub)} rows, sha256 {hashlib.sha256(open(out, 'rb').read()).hexdigest()}, "
          f"spec lane A SB3 + RC1 at w={w:.6f} cell {shown}, lane B from {parent}, floor 0")


if __name__ == "__main__":
    main(sys.argv[1] if sys.argv[1:] else "data/merry-octopus_v17.parquet",
         sys.argv[2] if sys.argv[2:] else PARENT)
