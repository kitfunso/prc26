"""Cache LT2's holdout prediction vector so MS1 can read the fourth controlled pair.

LT2 published a holdout RMSE (H64) but never wrote its per-row vector, so the fourth controlled pair is the one point
MS1 is missing. This refits LT2's searched cell on the ten training months and predicts Jan + Jul, exactly as `taxi.lt2`
reads it: `fit` returns the correction `c`, and the plan gap is added back before flooring.
"""
import optuna
import pandas as pd

from taxi import lt2 as _lt2
from taxi.champ import lane_a
from taxi.l2 import floored
from taxi.verdict import rmse

tra, tea = lane_a()
win = _lt2.cell(optuna.trial.FixedTrial(optuna.load_study(study_name=_lt2.STUDY, storage=f"sqlite:///{_lt2.DB}").best_params))
print(f"cell {win}", flush=True)
p = floored(_lt2.fit(win, tra, tea) + tea.z_aobt.values)
print(f"LT2 holdout RMSE {rmse(p - tea.y.values):.4f} on {len(p):,} rows (H64 published 232.97 for LT2 alone)", flush=True)
pd.DataFrame({"id": tea.id.values, "pred": p}).to_parquet("data/cache/lt2_holdout.parquet", index=False)
print("wrote data/cache/lt2_holdout.parquet", flush=True)
