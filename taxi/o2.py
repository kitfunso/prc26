"""O2: the joint parameter search. TPE over CatBoost's GPU-legal cell AND the recipe constants, on D2's columns.

Replaces the stale O1 draft in `docs/COVERAGE.md` section 6, which names LightGBM knobs from when M1 was champion.
HP1 walks one knob at a time off a hand-written list; this lane sees interactions and the knobs that list never reaches.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Callable

import numpy as np
import optuna
import pandas as pd

from taxi.champ import COLS_D2, lane_a
from taxi.data import CV_FOLDS
from taxi.l2 import floored
from taxi.m2 import is_tail
from taxi.mc1 import CELL, cv_a, predict_cb, stages_cb
from taxi.sb3 import SEEDS
from taxi.searches import reap
from taxi.sb3 import predict as sb3_predict
from taxi.verdict import paired_delta, rmse, score
from taxi.w1 import verdict_line

FOLDS = (CV_FOLDS[0], CV_FOLDS[2], CV_FOLDS[4])  # declared subset: Feb/Mar, Jun/Aug, Nov/Dec, so one trial spans both seasons
ITER, TRIALS, TIMEOUT = 8000, 40, 56 * 3600  # 73 min/trial READ off the study on 2026-09-10, not the 40 the first four
# trials showed: 14 of 35 trials never completed and their wall clock is invisible to a mean over completions.
STORAGE, CACHE = "sqlite:///data/cache/o2.db", "data/cache/o2"
BASE_CV = 211.8  # docs/D2.md over five folds; trial 0 re-reads D2's cell on THIS objective so the search carries its own base


def space(t: optuna.Trial) -> tuple[dict, float, int]:
    """The GPU-legal CatBoost cell plus the two recipe constants that bind; an illegal pair is never suggested."""
    p = dict(depth=t.suggest_int("depth", 4, 10),
             learning_rate=t.suggest_float("learning_rate", 0.02, 0.2, log=True),
             l2_leaf_reg=t.suggest_float("l2_leaf_reg", 0.5, 50.0, log=True),
             random_strength=t.suggest_float("random_strength", 0.0, 10.0),
             border_count=t.suggest_categorical("border_count", [32, 128, 254]),
             one_hot_max_size=t.suggest_categorical("one_hot_max_size", [2, 16, 64, 255]),
             grow_policy=t.suggest_categorical("grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"]),
             leaf_estimation_iterations=t.suggest_int("leaf_estimation_iterations", 1, 10))
    if t.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli"]) == "Bayesian":
        p.update(bootstrap_type="Bayesian", bagging_temperature=t.suggest_float("bagging_temperature", 0.0, 3.0))
    else:
        p.update(bootstrap_type="Bernoulli", subsample=t.suggest_float("subsample", 0.5, 1.0))
    if p["grow_policy"] != "SymmetricTree":
        p["min_data_in_leaf"] = t.suggest_int("min_data_in_leaf", 1, 500, log=True)
    if p["grow_policy"] == "Lossguide":
        p["max_leaves"] = t.suggest_int("max_leaves", 16, 64)
    return p, t.suggest_categorical("tail", [400.0, 600.0, 900.0, 1200.0]), t.suggest_int("patience", 50, 300, log=True)


def fold_sq(tra: pd.DataFrame, fold: tuple[int, int], over: dict, tail: float, patience: int) -> tuple[float, int]:
    m = tra.month.isin(fold).values
    f, va = tra[~m], tra[m]
    models = stages_cb(f, va=va, iters=(ITER,) * 3, cols=COLS_D2, over=over, tail=tail, patience=patience)
    e = predict_cb(models, va, COLS_D2)[0] - va.y.values
    return float(e @ e), len(e)


def objective(tra: pd.DataFrame) -> Callable[[optuna.Trial], float]:
    def run(t: optuna.Trial) -> float:
        over, tail, patience = space(t)
        sq, n, t0 = 0.0, 0, time.time()
        for i, fold in enumerate(FOLDS):
            try:
                s, k = fold_sq(tra, fold, over, tail, patience)
            except Exception as exc:
                t.set_user_attr("error", f"{type(exc).__name__}: {exc}"[:200])
                raise optuna.TrialPruned()  # an illegal GPU pair is recorded as a pruned trial, never a crashed study
            sq, n = sq + s, n + k
            t.report(float(np.sqrt(sq / n)), i)
            if t.should_prune():
                raise optuna.TrialPruned()
        t.set_user_attr("min", (time.time() - t0) / 60)
        print(f"trial {t.number}: {np.sqrt(sq / n):.2f} s in {(time.time() - t0) / 60:.0f} min "
              f"{json.dumps(over)} tail={tail} pat={patience}", flush=True)
        return float(np.sqrt(sq / n))
    return run


def bag(tra: pd.DataFrame, tea: pd.DataFrame, over: dict, tail: float, patience: int, rounds: tuple[int, int, int]) -> np.ndarray:
    key = hashlib.sha256(json.dumps([over, tail, patience], sort_keys=True).encode()).hexdigest()[:10]
    preds = []
    for s in SEEDS:
        path = f"{CACHE}_{key}_seed{s}.parquet"
        if os.path.exists(path):
            c = pd.read_parquet(path)
            if len(c) == len(tea) and np.array_equal(c.id.values, tea.id.values):
                preds.append(c.pred.values)
                continue
        t0 = time.time()
        m = stages_cb(tra, iters=rounds, cols=COLS_D2, over={**over, "random_seed": s}, tail=tail, patience=patience)
        p = predict_cb(m, tea, COLS_D2)[0]
        pd.DataFrame({"id": tea.id.values, "pred": p}).to_parquet(path, index=False)
        print(f"seed {s}: {time.time() - t0:.0f} s{score(p, tea)}", flush=True)
        preds.append(p)
    return floored(np.mean(preds, axis=0))


def base_trial() -> dict:
    """D2's own cell as trial 0, so the search measures its base on the same three folds it optimises."""
    return dict(depth=CELL["depth"], learning_rate=CELL["learning_rate"], l2_leaf_reg=CELL["l2_leaf_reg"],
                border_count=CELL["border_count"], random_strength=1.0, one_hot_max_size=2, grow_policy="SymmetricTree",
                leaf_estimation_iterations=1, bootstrap_type="Bayesian", bagging_temperature=1.0, tail=600.0, patience=100)


def main(out: str = "docs/O2.md") -> None:
    tra, tea = lane_a()
    os.makedirs("data/cache", exist_ok=True)
    study = optuna.create_study(study_name="o2", storage=STORAGE, load_if_exists=True, direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=0),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1))
    if not study.trials:
        study.enqueue_trial(base_trial())
    dead = reap(study)
    if dead:
        print(f"reaped {dead} trials left RUNNING by a killed process", flush=True)

    def enough(st: optuna.Study, _: optuna.trial.FrozenTrial) -> None:
        if len(st.get_trials(states=(optuna.trial.TrialState.COMPLETE,), deepcopy=False)) >= TRIALS:
            st.stop()  # an illegal GPU pair costs seconds, so budget the COMPLETED trials rather than the attempts

    study.optimize(objective(tra), n_trials=None, timeout=TIMEOUT, gc_after_trial=True, callbacks=[enough])

    ok = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state != optuna.trial.TrialState.COMPLETE]
    k, best = len(ok), study.best_trial
    base = next((t.value for t in ok if t.number == 0), None)
    rows = [f"| {t.number} | {t.value:.2f} | `{json.dumps(t.params)}` | {t.user_attrs.get('min', 0):.0f} |"
            for t in sorted(ok, key=lambda t: t.value)[:12]]

    lines = ["# O2: joint parameter search, Jan + Jul 2025 development holdout", "",
             f"Regenerate with `python -m taxi.o2`. Optuna TPE, seed 0, {TRIALS} trials or {TIMEOUT / 3600:.0f} h, median pruner "
             f"(8 startup trials, 1 warm-up fold); k = {k} completed trials. Objective: pooled RMSE over the three declared folds "
             f"{FOLDS} on D2's {len(COLS_D2)} columns, up to {ITER} iterations so a low learning rate is never censored (HP1's lr05 "
             f"hit the 5,000 cap on three of five folds). Trial 0 is D2's own cell on this objective, so the base is measured here "
             f"rather than carried across. Search space: depth, learning_rate, l2_leaf_reg, random_strength, border_count, "
             f"one_hot_max_size, grow_policy, leaf_estimation_iterations, bootstrap_type with its own bagging_temperature or "
             f"subsample, min_data_in_leaf and max_leaves where the grow policy allows them, plus the two recipe constants "
             f"`taxi.m2.TAIL` and the early-stopping patience.", "",
             f"{len(study.trials)} trials: {k} complete, {len(pruned)} pruned or illegal.",
             (f"Trial 0, D2's cell on this objective: {base:.2f} s." if base is not None else ""), "",
             "## Best twelve trials", "", "| trial | objective (s) | params | min |", "|---|---|---|---|", *rows, ""]

    if base is None or best.value >= base:
        lines += [f"Best trial {best.number} at {best.value:.2f} s does not beat D2's own cell on the same objective, so nothing "
                  f"meets the holdout and SB3 stands. The cell and the two recipe constants are searched jointly and closed."]
    else:
        over, tail, patience = space(optuna.trial.FixedTrial(best.params))
        c = cv_a(tra, COLS_D2, "O2-best", ref="D2", over=over, tail=tail, patience=patience, iters=(ITER,) * 3)
        y = tea.y.values
        e0, e1 = sb3_predict(tra, tea) - y, bag(tra, tea, over, tail, patience, tuple(c.rounds)) - y
        v = paired_delta(e0, e1, tea.day.values)
        vt, jan, jul = is_tail(tea), tea.month.values == 1, tea.month.values == 7
        row = lambda n, e: f"| {n} | {rmse(e):.2f} | {rmse(e[jan]):.1f} | {rmse(e[jul]):.1f} | {rmse(e[vt]):.1f} | {rmse(e[~vt]):.1f} |"  # noqa: E731
        lines += [f"Best trial {best.number} beats trial 0 by {base - best.value:.2f} s on the three-fold objective. Refit over all "
                  f"five folds it reads {c.rmse:.1f} s against D2's {BASE_CV} s; SB3 is a five-seed bag, so the challenger is bagged "
                  f"the same way.", "", *c.lines, "",
                  f"## Holdout verdict vs SB3 ({len(tea):,} rows)", "",
                  "| model | RMSE | Jan | Jul | tail rows | body rows |", "|---|---|---|---|---|---|",
                  row("SB3", e0), row("O2-best bag", e1), "", verdict_line("O2-best", "SB3", v, k=k)]

    text = "\n".join(lines) + "\n"
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
