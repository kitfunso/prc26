"""Optuna over every hyperparameter of one lane-A blend member, tuned on one fold set and judged on a disjoint one.
`python tune.py 200 ra_cb_wz 2-3,4-5 6-8,9-10,11-12`. The fold split keys the study, because a score averaged over one
fold set is not comparable with another. Trial 0 is the shipped params, so the search is always judged against the
thing it must beat. Tree members keep their shipped classifier, precomputed once per fold. The holdout is never touched."""
from __future__ import annotations

import sys
import time

import numpy as np
import optuna

import nn as nnmod
from data import rmse
from features import PW, build, with_pw
from lanea import CONFIGS, TAIL, _c
from mix import fit_clf
from train import HOLDOUT, fit_predict

# Every shipped value, including the library defaults the configs never set. Each one sits inside its range below.
SHIPPED = {
    "lgb": dict(lr=0.05, leaves=255, mcs=50, mcw=1e-3, ff=0.8, bf=0.8, bfreq=1, l1=0.0, l2=3.0, msg=0.0,
                maxbin=255, psmooth=0.0, extra=False, catsm=10.0, catl2=10.0, maxcat=32, mdpg=100),
    "cb": dict(depth=8, lr=0.1, l2=3.0, rs=1.0, bc=254, ctr=4, ohe=2, msr=0.5, lei=1, lem="Newton",
               grow="SymmetricTree", boot="Bayesian", bt=1.0),
    "nn": dict(layers=3, width=1024, dropout=0.2555, lr=1.5e-3, wd=1.65e-4, batch=1024, tmax=60,
               embcap=16, huber=0.405, patience=5, act="SiLU", bn=True),
}


def space(fam: str, t: optuna.Trial, base: dict) -> dict:
    """Iteration counts are absent on purpose: early stopping picks them, so searching them would fight the callback."""
    if fam == "lgb":
        return {**base,
                "learning_rate": t.suggest_float("lr", 0.01, 0.15, log=True),
                "num_leaves": t.suggest_int("leaves", 31, 1023, log=True),
                "min_child_samples": t.suggest_int("mcs", 5, 300, log=True),
                "min_child_weight": t.suggest_float("mcw", 1e-5, 10.0, log=True),
                "feature_fraction": t.suggest_float("ff", 0.4, 1.0),
                "bagging_fraction": t.suggest_float("bf", 0.4, 1.0),
                "bagging_freq": t.suggest_int("bfreq", 0, 7),
                "lambda_l1": t.suggest_float("l1", 0.0, 10.0),
                "lambda_l2": t.suggest_float("l2", 0.0, 50.0),
                "min_split_gain": t.suggest_float("msg", 0.0, 1.0),
                "max_bin": t.suggest_int("maxbin", 63, 511, log=True),
                "path_smooth": t.suggest_float("psmooth", 0.0, 100.0),
                "extra_trees": t.suggest_categorical("extra", [False, True]),
                "cat_smooth": t.suggest_float("catsm", 1.0, 200.0, log=True),
                "cat_l2": t.suggest_float("catl2", 1.0, 200.0, log=True),
                "max_cat_threshold": t.suggest_int("maxcat", 8, 128, log=True),
                "min_data_per_group": t.suggest_int("mdpg", 10, 500, log=True)}
    if fam == "cb":
        p = {**base,
             "depth": t.suggest_int("depth", 4, 10),
             "learning_rate": t.suggest_float("lr", 0.01, 0.2, log=True),
             "l2_leaf_reg": t.suggest_float("l2", 0.5, 30.0, log=True),
             "random_strength": t.suggest_float("rs", 0.0, 10.0),
             "border_count": t.suggest_int("bc", 32, 254),
             "max_ctr_complexity": t.suggest_int("ctr", 1, 4),
             "one_hot_max_size": t.suggest_int("ohe", 2, 255, log=True),
             "model_size_reg": t.suggest_float("msr", 0.0, 2.0),
             "leaf_estimation_iterations": t.suggest_int("lei", 1, 10),
             "leaf_estimation_method": t.suggest_categorical("lem", ["Newton", "Gradient"]),
             "grow_policy": t.suggest_categorical("grow", ["SymmetricTree", "Depthwise", "Lossguide"]),
             "bootstrap_type": t.suggest_categorical("boot", ["Bayesian", "Bernoulli", "No"])}
        if p["bootstrap_type"] == "Bayesian":
            p["bagging_temperature"] = t.suggest_float("bt", 0.0, 5.0)
        elif p["bootstrap_type"] == "Bernoulli":
            p["subsample"] = t.suggest_float("sub", 0.3, 1.0)
        if p["grow_policy"] != "SymmetricTree":
            p["min_data_in_leaf"] = t.suggest_int("mdl", 1, 200, log=True)
        if p["grow_policy"] == "Lossguide":
            p["max_leaves"] = t.suggest_int("mleaves", 16, 128, log=True)
        return p
    layers = t.suggest_int("layers", 1, 5)
    width = t.suggest_int("width", 128, 2048, log=True)
    return {**base, "hidden": (width,) * layers,
            "dropout": t.suggest_float("dropout", 0.0, 0.6),
            "lr": t.suggest_float("lr", 1e-4, 1e-2, log=True),
            "weight_decay": t.suggest_float("wd", 1e-7, 1e-2, log=True),
            "batch": t.suggest_categorical("batch", [256, 512, 1024, 2048, 4096]),
            "t_max": t.suggest_int("tmax", 10, 150, log=True),
            "emb_cap": t.suggest_int("embcap", 4, 64, log=True),
            "huber": t.suggest_float("huber", 0.05, 3.0, log=True),
            "patience": t.suggest_int("patience", 3, 15),
            "act": t.suggest_categorical("act", ["SiLU", "ReLU", "GELU"]),
            "bn": t.suggest_categorical("bn", [True, False])}


def score(cfg: dict, fam: str, params: dict, tr, trc, tail, folds: list, t: optuna.Trial | None = None) -> float:
    out = []
    for i, (va, P) in enumerate(folds):
        if fam == "nn":
            p, _ = nnmod.predict(dict(cfg, cell=params), tr[~va], tr[va])
            out.append(rmse(tr["y"].values[va], p))
        else:
            reg = dict(cfg["reg"], params=params)
            a, b, ta, tb = trc[~va], trc[va], tail[~va], tail[va]
            ct, _ = fit_predict(reg, a[ta], b, b[tb])
            cb, _ = fit_predict(reg, a[~ta], b, b[~tb])
            out.append(rmse(b["y"].values, P * ct + (1 - P) * cb))
        if t is not None:
            t.report(float(np.mean(out)), i)
            if t.should_prune():
                raise optuna.TrialPruned()
    return float(np.mean(out))


def main(n_trials: int, name: str, tune_f: tuple, judge_f: tuple) -> None:
    is_nn = name.startswith("nn")
    cfg = nnmod.CONFIGS[name] if is_nn else CONFIGS[name]
    fam = "nn" if is_nn else cfg["reg"].get("family", "cb")
    base = cfg["cell"] if is_nn else cfg["reg"]["params"]
    f = build()
    if set(PW) & set(cfg["num"]):
        f = with_pw(f)
    tr = f[(f["split"] == "train") & ~f["month"].isin(HOLDOUT) & (f["has_plan"] == 1)]
    trc = _c(tr)
    tail = (trc["y"].abs() > TAIL).values
    fold = {}
    for m in tune_f + judge_f:
        va = tr["month"].isin(m).values
        P = None if is_nn else fit_clf(cfg, trc[~va], tail[~va].astype(int), trc[va], trc[va], tail[va].astype(int), None)[0][:, 1]
        fold[m] = (va, P)
        print(f"fold {m} ready, n_va={int(va.sum())}", flush=True)

    def run(params: dict, folds: tuple, t: optuna.Trial | None = None) -> float:
        return score(cfg, fam, params, tr, trc, tail, [fold[m] for m in folds], t)

    def objective(t: optuna.Trial) -> float:
        t0 = time.time()
        try:
            return run(space(fam, t, base), tune_f, t)
        finally:
            t.set_user_attr("secs", round(time.time() - t0))

    tag = "_".join("".join(str(x) for x in m) for m in tune_f)
    study = optuna.create_study(study_name=f"{name}_full_{tag}", storage="sqlite:///data/optuna.db", load_if_exists=True,
                                direction="minimize", sampler=optuna.samplers.TPESampler(seed=0),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=15, n_warmup_steps=0))
    if not any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
        study.enqueue_trial(SHIPPED[fam])
    study.optimize(objective, n_trials=n_trials, catch=(Exception,))
    n = {s: sum(1 for t in study.trials if t.state.name == s) for s in ("COMPLETE", "PRUNED", "FAIL")}
    print(f"\n{n} on tune folds {tune_f}: best {study.best_value:.3f} at trial {study.best_trial.number}", flush=True)
    print(f"best params: {study.best_params}", flush=True)
    b0 = run(base, judge_f)
    b1 = run(space(fam, optuna.trial.FixedTrial(study.best_params), base), judge_f)
    print(f"JUDGE folds {judge_f}, never searched: shipped {b0:.3f}, tuned {b1:.3f}, gain {b0 - b1:+.3f}", flush=True)


if __name__ == "__main__":
    p = lambda s: tuple(tuple(int(x) for x in g.split("-")) for g in s.split(","))
    main(int(sys.argv[1]), sys.argv[2], p(sys.argv[3]), p(sys.argv[4]))
