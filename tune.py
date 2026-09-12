"""Optuna over the lane-A residual regressors (tail and body share one CatBoost GPU param set; classifier fixed) on CV folds; holdout untouched.
`python tune.py N_TRIALS [FOLDS_CSV]`, study `ra_reg` in data/optuna.db; trial 0 is the ra_cb baseline on the same folds."""
from __future__ import annotations

import sys
import time

import numpy as np
import optuna

from data import rmse
from features import CAT, DAY, NUM, OCC, WX, build
from lanea import CONFIGS, predict
from train import BASE, FOLDS, HOLDOUT

BASELINE = dict(depth=8, lr=0.1, l2=3, rs=1, bt=1, bc=254, ctr=4, ohe=2)


def main(n_trials: int, folds: tuple[tuple[int, int], ...]) -> None:
    f = build()
    tr = f[(f["split"] == "train") & ~f["month"].isin(HOLDOUT) & (f["has_plan"] == 1)]
    masks = [tr["month"].isin(m).values for m in folds]
    cols = NUM + DAY + OCC + WX

    def objective(t: optuna.Trial) -> float:
        params = dict(BASE, thread_count=8, gpu_ram_part=0.5, depth=t.suggest_int("depth", 6, 10), learning_rate=t.suggest_float("lr", 0.03, 0.15, log=True),
                      l2_leaf_reg=t.suggest_float("l2", 1, 30, log=True), random_strength=t.suggest_float("rs", 0.3, 5, log=True),
                      bagging_temperature=t.suggest_float("bt", 0, 1), border_count=t.suggest_categorical("bc", [128, 254]),
                      max_ctr_complexity=t.suggest_categorical("ctr", [1, 2, 4]), one_hot_max_size=t.suggest_categorical("ohe", [2, 10]))
        cfg = dict(CONFIGS["ra_cb"], reg=dict(params=params, cat=CAT, num=cols))
        scores, t0 = [], time.time()
        for i, va in enumerate(masks):
            p, it = predict(cfg, tr[~va], tr[va], tr[va])
            scores.append(rmse(tr["y"].values[va], p))
            t.set_user_attr(f"it{i}", it)
            t.report(float(np.mean(scores)), i)
            if t.should_prune():
                raise optuna.TrialPruned()
        t.set_user_attr("secs", round(time.time() - t0))
        return float(np.mean(scores))

    study = optuna.create_study(study_name="ra_reg", storage="sqlite:///data/optuna.db", load_if_exists=True, direction="minimize",
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=0))
    study.enqueue_trial(BASELINE) if not study.trials else None
    study.optimize(objective, n_trials=n_trials)
    print("best", study.best_value, study.best_params, study.best_trial.user_attrs)


if __name__ == "__main__":
    fs = tuple(tuple(int(x) for x in p.split("-")) for p in sys.argv[2].split(",")) if len(sys.argv) > 2 else FOLDS
    main(int(sys.argv[1]), fs)
