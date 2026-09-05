"""O1: Optuna TPE over MC2's cell (six axes); objective = five-fold month-pair CV RMSE in MC2's recipe on its columns; the best trial's cell refit at its
CV rounds meets the holdout once, k = trials completed."""
from __future__ import annotations

import time

import numpy as np
import optuna

from taxi.champ import COLS_D1, champion_holdout, lane_a
from taxi.m2 import is_tail
from taxi.mc1 import cv_a, predict_cb, stages_cb
from taxi.verdict import paired_delta, rmse
from taxi.w1 import verdict_line

STORAGE, STUDY = "sqlite:///data/o1.db", "o1"
N_TRIALS, TIMEOUT = 40, 24 * 3600
MC2_CELL = dict(learning_rate=0.1, depth=8, l2_leaf_reg=3.0, border_count=254, random_strength=1.0, bagging_temperature=1.0)


def cell(trial: optuna.Trial) -> dict:
    return dict(learning_rate=trial.suggest_float("learning_rate", 0.02, 0.2, log=True), depth=trial.suggest_int("depth", 4, 10),
                l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 0.1, 30.0, log=True), border_count=trial.suggest_categorical("border_count", [128, 254]),
                random_strength=trial.suggest_float("random_strength", 0.0, 5.0), bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 2.0),
                bootstrap_type="Bayesian")


def objective(trial: optuna.Trial, tra) -> float:
    t0 = time.time()
    r = cv_a(tra, COLS_D1, f"O1-{trial.number}", ref="MC2", over=cell(trial))
    trial.set_user_attr("rounds", list(r.rounds))
    trial.set_user_attr("minutes", (time.time() - t0) / 60)
    print(f"trial {trial.number}: CV {r.rmse:.2f} s, rounds {r.rounds}, {trial.user_attrs['minutes']:.0f} min, {trial.params}", flush=True)
    return r.rmse


def main() -> None:
    tra, tea = lane_a()
    study = optuna.create_study(study_name=STUDY, storage=STORAGE, direction="minimize", sampler=optuna.samplers.TPESampler(seed=0), load_if_exists=True)
    if not study.trials:
        study.enqueue_trial(MC2_CELL)  # trial 0 = the incumbent, so the study carries MC2's own CV on the same data
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(done) < N_TRIALS:
        study.optimize(lambda t: objective(t, tra), n_trials=N_TRIALS - len(done), timeout=TIMEOUT, catch=(Exception,))
    done = sorted((t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE), key=lambda t: t.number)
    best, k, y = study.best_trial, len(done), tea.y.values
    over = {**best.params, "bootstrap_type": "Bayesian"}
    rounds = tuple(best.user_attrs["rounds"])
    t0 = time.time()
    p1 = predict_cb(stages_cb(tra, iters=rounds, cols=COLS_D1, over=over), tea, COLS_D1)[0]
    fit_min = (time.time() - t0) / 60
    e0, e1 = champion_holdout("MC2", tra, tea) - y, p1 - y
    v = paired_delta(e0, e1, tea.day.values)
    vt, jan, jul = is_tail(tea), tea.month.values == 1, tea.month.values == 7
    row = lambda name, e: f"| {name} | {rmse(e):.1f} | {rmse(e[jan]):.1f} | {rmse(e[jul]):.1f} | {rmse(e[vt]):.1f} | {rmse(e[~vt]):.1f} |"  # noqa: E731
    trials = [f"| {t.number} | {t.value:.2f} | {tuple(t.user_attrs['rounds'])} | {t.user_attrs['minutes']:.0f} | " +
              ", ".join(f"{a} {b:.3g}" for a, b in t.params.items()) + " |" for t in done]
    failed = [t.number for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    report = ["# O1: Optuna TPE over MC2's cell, Jan+Jul 2025 development holdout", "",
              f"Regenerate with `python -m taxi.o1` (the study lives in `{STORAGE}` and resumes). Sampler TPE seed 0; trial 0 is MC2's own cell "
              f"(the incumbent on the same data); axes: learning rate log [0.02, 0.2], depth [4, 10], l2 log [0.1, 30], border count {{128, 254}}, "
              f"random strength [0, 5], bagging temperature [0, 2] (Bayesian bootstrap). Objective: five-fold month-pair CV RMSE in MC2's recipe on its "
              f"{len(COLS_D1)} columns with early stopping. Budget {N_TRIALS} trials or {TIMEOUT // 3600} h; {k} completed, failed trials {failed or 'none'}. "
              f"TPE trials are not independent draws, so the k = {k} bound is a floor on the optimism (reviewer condition 5). "
              f"A resumed study restarts the sampler's seed, so the trial sequence is reproducible only from an uninterrupted run.", "",
              "## Trials", "", "| trial | CV RMSE (s) | rounds (clf, tail, body) | min | cell |", "|---|---|---|---|---|", *trials, "",
              f"Best trial {best.number}: CV {best.value:.2f} s against trial 0 (MC2's cell) {done[0].value:.2f} s.", "",
              f"## Holdout verdict vs champion MC2 ({len(tea)} rows; tail rows {int(vt.sum())}; best cell refit at rounds {rounds}, {fit_min:.1f} min)", "",
              "| model | RMSE | Jan | Jul | tail rows | body rows |", "|---|---|---|---|---|---|", row("MC2", e0), row(f"O1 (trial {best.number})", e1), "",
              verdict_line("O1", "MC2", v, k=k)]
    text = "\n".join(report) + "\n"
    with open("docs/O1.md", "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
