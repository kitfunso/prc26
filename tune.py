"""Optuna over the lane-A residual regressors (tail and body share one CatBoost param set; classifier fixed and precomputed) on CV folds; holdout untouched.
`python tune.py N_TRIALS [FOLDS_CSV] [--cpu]`, study `ra_reg` in data/optuna.db; trial 0 is the ra_cb baseline on the same folds."""
from __future__ import annotations

import sys
import time

import numpy as np
import optuna

from data import rmse
from features import CAT, DAY, NUM, OCC, WX, build
from lanea import CONFIGS, TAIL, _c
from mix import fit_clf
from train import BASE, FOLDS, HOLDOUT, fit_predict

BASELINE = dict(depth=8, lr=0.1, l2=3, rs=1, bt=1, bc=254, ctr=4, ohe=2)


DEV = "CPU" if "--cpu" in sys.argv else "GPU"


def dev(p: dict) -> dict:
    """--cpu keeps the tuner off a shared GPU; measured 2026-09-12 at about 1 h per trial on 16 threads, so it is the fallback only."""
    return {**{k: v for k, v in p.items() if k != "gpu_ram_part"}, "task_type": "CPU", "thread_count": 16} if DEV == "CPU" else dict(p, thread_count=8, gpu_ram_part=0.5)


def main(n_trials: int, folds: tuple[tuple[int, int], ...]) -> None:
    f = build()
    tr = f[(f["split"] == "train") & ~f["month"].isin(HOLDOUT) & (f["has_plan"] == 1)]
    masks = [tr["month"].isin(m).values for m in folds]
    cols = NUM + DAY + OCC + WX
    tr = _c(tr); tail = (tr["y"].abs() > TAIL).values
    clf = dict(CONFIGS["ra_cb"], clf=dict(family="cb", params=dev(CONFIGS["ra_cb"]["clf"]["params"])))
    P = [fit_clf(clf, tr[~va], tail[~va].astype(int), tr[va], tr[va], tail[va].astype(int), None)[0][:, 1] for va in masks]
    print("classifier done", flush=True)

    def objective(t: optuna.Trial) -> float:
        params = dict(dev(BASE), depth=t.suggest_int("depth", 6, 10), learning_rate=t.suggest_float("lr", 0.03, 0.15, log=True),
                      l2_leaf_reg=t.suggest_float("l2", 1, 30, log=True), random_strength=t.suggest_float("rs", 0.3, 5, log=True),
                      bagging_temperature=t.suggest_float("bt", 0, 1), border_count=t.suggest_categorical("bc", [128, 254]),
                      max_ctr_complexity=t.suggest_categorical("ctr", [1, 2, 4]), one_hot_max_size=t.suggest_categorical("ohe", [2, 10]))
        reg = dict(params=params, cat=CAT, num=cols)
        scores, t0 = [], time.time()
        for i, va in enumerate(masks):
            a, b, ta, tb = tr[~va], tr[va], tail[~va], tail[va]
            ct, i1 = fit_predict(reg, a[ta], b, b[tb])
            cb, i2 = fit_predict(reg, a[~ta], b, b[~tb])
            scores.append(rmse(b["y"].values, P[i] * ct + (1 - P[i]) * cb))
            t.set_user_attr(f"it{i}", (i1, i2))
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
    args = [a for a in sys.argv[1:] if a != "--cpu"]
    fs = tuple(tuple(int(x) for x in p.split("-")) for p in args[1].split(",")) if len(args) > 1 else FOLDS
    main(int(args[0]), fs)
