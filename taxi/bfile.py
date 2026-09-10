"""BFILE: the ranking file with lane A blended SB3 + one partner, lane B taken from the parent unchanged.

Generalises `taxi.rc1s`, which named RC1's fit, store and weight inline and so offered no entry point for a second partner.
SB3's lane-A values are read from the parent file by id rather than refit, because GPU CatBoost fits do not reproduce (S3, S4),
so the blend is the only new arithmetic in the file. Weights are PINNED to the published number rather than recomputed: each
file's claimed holdout RMSE was read at that exact weight, and a recomputed one would ship a blend no verdict ever saw. Every
weight here was fitted out of fold against MC2, not SB3, which is the open finding `taxi.sb3oof` exists to close.
Build only; upload is a Keith gate.
"""
from __future__ import annotations

import hashlib
import sys

import numpy as np
import optuna
import pandas as pd

from taxi import lt2 as _lt2
from taxi import nn3 as _nn3
from taxi.data import HOLDOUT_MONTHS, TRAIN_MONTHS, features, load
from taxi.l2 import floored
from taxi.nn1 import predict as nn_predict
from taxi.nn1 import train as nn_train
from taxi.nn2 import PATIENCE
from taxi.s2 import RANKING
from taxi.sb3 import SEEDS
from taxi.submit import build_submission
from taxi.weather import join, load_metar

PARENT = "data/merry-octopus_v17.parquet"


def _study_cell(module, name: str, storage: str):
    return module(optuna.trial.FixedTrial(optuna.load_study(study_name=name, storage=storage).best_params))


def _nn_arm(cell_fn, patience: int | None):
    """A bagged net over `taxi.sb3.SEEDS`. `nn1.predict` already adds the plan gap back, so its output is a level."""
    def go(tr: pd.DataFrame, rk: pd.DataFrame) -> np.ndarray:
        cell, preds = cell_fn(), []
        for s in SEEDS:
            model, enc, mu, sd, ep = nn_train(tr, seed=s, patience=patience, cell=cell)
            preds.append(nn_predict(model, enc, rk, mu, sd))
            print(f"  seed {s}: {ep} epochs, mean {preds[-1].mean():.1f} s", flush=True)
        return floored(np.mean(preds, axis=0))
    return go


def _lt2_arm(tr: pd.DataFrame, rk: pd.DataFrame) -> np.ndarray:
    """`lt2.fit` predicts the correction `c`, so the plan gap is added back here exactly as `lt2.blended` does."""
    win = _study_cell(_lt2.cell, _lt2.STUDY, f"sqlite:///{_lt2.DB}")
    return floored(_lt2.fit(win, tr, rk) + rk.z_aobt.values)


PARTNERS = {
    "NN3": (0.5189, _nn_arm(lambda: _study_cell(_nn3.suggest, "nn3", _nn3.STORE), None),
            "the 13-knob searched net, 5-seed bag, weight from `docs/NN3.md`"),
    "NN2": (0.4160, _nn_arm(lambda: None, PATIENCE),
            "NN1's cell at patience 15, 5-seed bag, weight from ledger H59"),
    "LT2": (0.5307, _lt2_arm,
            "LightGBM linear_tree at its searched cell, weight from `docs/LT2.md`"),
}


def main(out: str, partner: str, parent: str = PARENT) -> None:
    w, arm, spec = PARTNERS[partner]
    metar = load_metar()
    tr, rk = (join(d, metar) for d in (load(TRAIN_MONTHS + HOLDOUT_MONTHS), features([RANKING])))
    m = rk.has_fpl.values
    print(f"{partner}: fitting on {len(tr):,} rows, predicting {int(m.sum()):,} lane-A ranking rows", flush=True)
    part = arm(tr[tr.has_fpl].reset_index(drop=True), rk[m].reset_index(drop=True))

    pp = pd.read_parquet(parent)
    par = pd.Series(pp.TAXITIME_SEC_mvt.values.astype("float64"), index=pp.MVT_ID_mvt.values).reindex(rk.id.values)
    if par.isna().any():
        raise ValueError(f"{int(par.isna().sum())} ranking ids missing from {parent}")
    vals, sb3 = par.to_numpy(float).copy(), par.to_numpy(float)[m]
    vals[m] = floored(w * part + (1 - w) * sb3)
    sub = build_submission(pd.Series(vals, index=rk.id.values), out, floor=0.0)
    print(f"lane A: {int(m.sum())} rows, w {w:.4f}, {partner} mean {part.mean():.1f}, SB3 mean {sb3.mean():.1f}, "
          f"blend mean {vals[m].mean():.1f}, rows moved by 1 s or more {int((np.abs(vals[m] - sb3) >= 1).sum())}, "
          f"largest move {np.abs(vals[m] - sb3).max():.1f} s", flush=True)
    print(f"lane B: {int((~m).sum())} rows from {parent}, unchanged")
    print(f"{out}: {len(sub)} rows, sha256 {hashlib.sha256(open(out, 'rb').read()).hexdigest()}, "
          f"spec lane A SB3 + {partner} at w={w:.4f} ({spec}), lane B from {parent}, floor 0")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if sys.argv[3:] else PARENT)
