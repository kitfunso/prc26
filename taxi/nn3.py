"""NN3 (protocol section 4): the first parameter search over the neural net, 13 knobs, judged on the BLEND not the net.

NN1 declared one cell and NN2 changed only patience and the seed bag, which alone bought 5.16 s. `docs/BF.md` re-run at NN2's
measured agreement says a partner at 233.0 s clears the 2 s floor, so the net needs 1.2 s and has never been searched. The
objective is the blended RMSE because level and agreement set the gain together and NN1 to NN2 moved them in opposite directions.
"""
from __future__ import annotations

import time

import numpy as np
import optuna
import pandas as pd

from taxi.bf import best_blend
from taxi.champ import COLS_D1, COLS_D2, lane_a
from taxi.l2 import floored
from taxi.nn1 import base_oof, blend_weight, predict, train
from taxi.nn2 import bagged_holdout, bagged_oof
from taxi.sb3 import predict as sb3_predict
from taxi.verdict import paired_delta, rmse
from taxi.w1 import verdict_line

FOLDS = ((2, 3), (6, 8), (11, 12))
TRIALS, BUDGET_S, WIDTHS = 40, 8 * 3600, (128, 256, 512, 768, 1024)
STORE, OOF_CACHE, TEA_CACHE = "sqlite:///data/cache/nn3.db", "data/cache/nn3_oof.parquet", "data/cache/nn3_holdout.parquet"


def suggest(t: optuna.Trial) -> dict:
    """All 13 knobs. Widths narrow down the stack so a trial cannot spend its budget on a widening tower."""
    n = t.suggest_int("layers", 2, 4)
    cap = len(WIDTHS) - 1
    hidden = []
    for i in range(n):
        cap = t.suggest_int(f"w{i}", 0, cap)
        hidden.append(WIDTHS[cap])
    ep = t.suggest_categorical("max_epochs", [40, 80, 150])
    return dict(
        hidden=tuple(hidden), dropout=t.suggest_float("dropout", 0.0, 0.4),
        lr=t.suggest_float("lr", 1e-4, 5e-3, log=True), weight_decay=t.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        batch=t.suggest_categorical("batch", [1024, 2048, 4096, 8192]), max_epochs=ep,
        t_max=min(t.suggest_categorical("t_max", [20, 40, 60, 100, 150]), ep),
        patience=t.suggest_categorical("patience", [5, 10, 20, 35]),
        emb_cap=t.suggest_categorical("emb_cap", [8, 16, 32, 64]),
        act=t.suggest_categorical("act", ["gelu", "relu", "silu"]),
        norm=t.suggest_categorical("norm", ["batch", "layer", "none"]),
        loss=t.suggest_categorical("loss", ["mse", "huber"]), huber=t.suggest_float("huber", 0.1, 2.0, log=True),
        cols=COLS_D2 if t.suggest_categorical("cols", ["d1", "d2"]) == "d2" else COLS_D1)


def score(cell: dict, tra: pd.DataFrame, base: np.ndarray, t: optuna.Trial | None = None) -> tuple[float, float, float]:
    """Pooled blend RMSE over the three search folds, with the net's own level and its agreement with the base beside it."""
    p, keep = np.full(len(tra), np.nan), np.zeros(len(tra), bool)
    for i, fold in enumerate(FOLDS):
        m = tra.month.isin(fold).values
        model, enc, mu, sd, _ = train(tra[~m].reset_index(drop=True), cell=cell)
        p[m], keep[m] = predict(model, enc, tra[m].reset_index(drop=True), mu, sd), True
        if t is not None:
            ok = keep & np.isfinite(base)
            t.report(rmse(floored(p[ok]) - tra.y.values[ok]), i)
            if t.should_prune():
                raise optuna.TrialPruned()
    ok = keep & np.isfinite(base)
    y, nn = tra.y.values[ok], floored(p[ok])
    w = blend_weight(nn, base[ok], y)
    return rmse(floored(w * nn + (1 - w) * base[ok]) - y), rmse(nn - y), best_blend(base[ok] - y, nn - y)[2]


def main(out: str = "docs/NN3.md") -> None:
    tra, tea = lane_a()
    base, t0, rows = base_oof(tra), time.time(), []

    def objective(t: optuna.Trial) -> float:
        if time.time() - t0 > BUDGET_S:
            t.study.stop()
        b, lvl, rho = score(suggest(t), tra, base, t)
        t.set_user_attr("level", lvl)
        t.set_user_attr("agreement", rho)
        rows.append(f"| {t.number} | {b:.2f} | {lvl:.2f} | {rho:.4f} |")
        print(rows[-1], flush=True)
        return b

    study = optuna.create_study(direction="minimize", storage=STORE, study_name="nn3", load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=0), pruner=optuna.pruners.MedianPruner(n_startup_trials=8))
    print("| trial | blend RMSE | net alone | agreement |\n|---|---|---|---|", flush=True)
    def enough(st: optuna.Study, _: optuna.trial.FrozenTrial) -> None:
        if len(st.get_trials(states=(optuna.trial.TrialState.COMPLETE,), deepcopy=False)) >= TRIALS:
            st.stop()  # n_trials counts ATTEMPTS while k below counts COMPLETIONS, so budget the completions

    study.optimize(objective, n_trials=None, timeout=BUDGET_S, callbacks=[enough])
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    k = len(done)
    # rebuilt from the study, not the live list, so a resumed run still writes every trial it was judged on
    rows = [f"| {t.number} | {t.value:.2f} | {t.user_attrs['level']:.2f} | {t.user_attrs['agreement']:.4f} |" for t in done]

    cell = suggest(optuna.trial.FixedTrial(study.best_params))
    oof, fold_lines = bagged_oof(tra, cell=cell, patience=None, cache=OOF_CACHE)
    ok = np.isfinite(oof) & np.isfinite(base)
    w = blend_weight(oof[ok], base[ok], tra.y.values[ok])
    nn_te = floored(bagged_holdout(tra, tea, cell=cell, patience=None, cache=TEA_CACHE))
    sb3_te, y = sb3_predict(tra, tea), tea.y.values
    e0, ex = sb3_te - y, nn_te - y
    v = paired_delta(e0, floored(w * nn_te + (1 - w) * sb3_te) - y, tea.day.values)
    ow, orm, rho = best_blend(e0, ex)

    shown = {n: (v if n != "cols" else ("d2" if v is COLS_D2 else "d1")) for n, v in cell.items()}
    text = "\n".join([
        "# NN3: the neural net searched on every parameter, lane A, Jan + Jul 2025 holdout", "",
        f"Regenerate with `python -m taxi.nn3`. Declared in protocol section 4 before the code was written. Optuna TPE, seed 0, "
        f"13 knobs, {k} completed trials of {TRIALS} budgeted in {(time.time() - t0) / 3600:.1f} h. The objective is the pooled "
        f"BLEND RMSE over folds {FOLDS} at a weight fitted on those rows, never the net's own level: `docs/BF.md` shows the gain "
        f"is set by level and agreement together, and NN1 to NN2 improved the level 5.16 s while agreement moved the wrong way.", "",
        "## Trials", "", "| trial | blend RMSE | net alone | agreement |", "|---|---|---|---|", *rows, "",
        f"Best cell: `{shown}`.", "",
        "## The best cell refit over five folds, bagged over five seeds", "", *fold_lines, "",
        f"Weight on NN3, fitted out of fold on {int(ok.sum()):,} rows: **{w:.4f}**. The holdout is never read to choose it.", "",
        f"## Holdout ({len(tea):,} rows)", "", "| model | RMSE |", "|---|---|",
        f"| SB3 (champion) | {rmse(e0):.2f} |", f"| NN3 alone (diagnostic) | {rmse(ex):.2f} |",
        f"| SB3 + NN3 at w={w:.4f} (challenger) | {rmse(floored(w * nn_te + (1 - w) * sb3_te) - y):.2f} |", "",
        f"NN3 reaches {rmse(ex):.2f} s at agreement **{rho:.4f}** with SB3, against NN2's 234.22 s at 0.9403 and NN1's 239.38 s "
        f"at 0.9169. The oracle weight puts {ow:.3f} on SB3 and reaches {orm:.2f} s, {rmse(e0) - orm:+.2f} s.", "",
        verdict_line("SB3 + NN3", "SB3", v, k=max(k, 1)), ""])
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
