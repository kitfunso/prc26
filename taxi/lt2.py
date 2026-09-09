"""LT2: tune FAM2's piecewise-linear tree on the objective its value actually lives on, the BLEND, not its own RMSE.

FAM2-lt was the only one of eight arms to pass its gate, and it passed while being 3.98 s WORSE alone than the base it
helps. Its whole value is the 0.9492 error agreement, so a search that minimised its own RMSE would walk it toward MC2's
function and spend the decorrelation that is the point. The Optuna objective here is therefore the blend RMSE against
MC2's cached out-of-fold at a weight fitted on the same rows. Declared in `docs/EXPERIMENT-PROTOCOL.md` section 4, row LT2.
"""
from __future__ import annotations

import gc
import sys
import time

import numpy as np
import pandas as pd

from taxi.champ import lane_a
from taxi.data import CV_FOLDS
from taxi.fam1 import FOLDS, _tree_block, oof
from taxi.l2 import floored
from taxi.nn1 import base_oof, blend_weight
from taxi.sb3 import predict as sb3_predict
from taxi.searches import SEARCHES, achieved, reap
from taxi.verdict import paired_delta, rmse
from taxi.w1 import verdict_line

TIMEOUT = 14 * 3600  # forced by the pre-launch timing probe below, not chosen: 40 trials x 21 min in the slowest corner
K = 40
DB, STUDY = "data/cache/lt2.db", "lt2"
HELD = tuple(f for f in CV_FOLDS if f not in FOLDS)  # months 4, 5, 9, 10: no cell here was ever ranked on them
GATE = 2.0  # read 1 carries the floor alone: cell and weight are both fixed by rows disjoint from every judged row, so k = 1


def cell(t) -> dict:
    """The six declared axes. `n_estimators` is TIED to the rate so the search cannot buy score with trees alone."""
    lr = t.suggest_float("learning_rate", 0.02, 0.2, log=True)
    return dict(learning_rate=lr, n_estimators=round(60 / lr),
                num_leaves=t.suggest_int("num_leaves", 16, 256, log=True),
                min_child_samples=t.suggest_int("min_child_samples", 20, 2000, log=True),
                linear_lambda=t.suggest_float("linear_lambda", 1e-3, 100.0, log=True),
                colsample_bytree=t.suggest_float("feature_fraction", 0.5, 1.0),
                reg_lambda=t.suggest_float("lambda_l2", 1e-3, 100.0, log=True))


# Timed 2026-09-09 BEFORE the search launched, at 100 trees then scaled to the declared 3000: 14.0 s a fold, 21 min a trial.
SLOWEST = dict(learning_rate=0.02, n_estimators=3000, num_leaves=256, min_child_samples=20,
               linear_lambda=1e-3, colsample_bytree=1.0, reg_lambda=1e-3)


def fit(params: dict, tr: pd.DataFrame, te: pd.DataFrame) -> np.ndarray:
    import lightgbm as lgb

    a, b = _tree_block(tr, te)
    m = lgb.LGBMRegressor(linear_tree=True, n_jobs=12, random_state=0, verbose=-1, **params).fit(a, tr.c.values)
    del a
    gc.collect()
    return m.predict(b)


def blended(tra: pd.DataFrame, p: np.ndarray, base: np.ndarray, mask: np.ndarray, w: float | None = None):
    """The objective and both reads share one assembly: floored levels, MC2's floored out-of-fold, one least-squares weight."""
    y = tra.y.values[mask]
    lvl, b = floored(p[mask] + tra.z_aobt.values[mask]), floored(base[mask])
    w = blend_weight(lvl, b, y) if w is None else w
    return rmse(floored(w * lvl + (1 - w) * b) - y), w, floored(w * lvl + (1 - w) * b) - y, b - y


def objective(t, tra: pd.DataFrame, base: np.ndarray) -> float:
    params = cell(t)
    p, _spent, _note = oof(tra, lambda tr, te: fit(params, tr, te), TIMEOUT)
    ok = np.isfinite(p) & np.isfinite(base)
    r, w, _, _ = blended(tra, p, base, ok)
    t.set_user_attr("w", w)
    return r


def search(tra: pd.DataFrame, base: np.ndarray):
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    st = optuna.create_study(study_name=STUDY, storage=f"sqlite:///{DB}", direction="minimize", load_if_exists=True,
                             sampler=optuna.samplers.TPESampler(seed=0))
    print(f"reaped {reap(st)} corpse trials", flush=True)
    done = len([x for x in st.trials if x.state == optuna.trial.TrialState.COMPLETE])
    if done < K:
        st.optimize(lambda t: objective(t, tra, base), n_trials=K - done, timeout=TIMEOUT,
                    callbacks=[lambda s, t: print(f"| {t.number} | {t.value if t.value else float('nan'):.4f} | "
                                                  f"{s.best_value:.4f} |", flush=True)])
    return st


def main(out: str = "docs/LT2.md") -> None:
    tra, tea = lane_a()
    base = base_oof(tra)
    t0 = time.time()
    print("| trial | blend RMSE | best |\n|---|---|---|", flush=True)
    st = search(tra, base)

    win = dict(st.best_params)
    win["n_estimators"], win["colsample_bytree"] = round(60 / win["learning_rate"]), win.pop("feature_fraction")
    win["reg_lambda"] = win.pop("lambda_l2")

    p5, _spent, _note = oof(tra, lambda tr, te: fit(win, tr, te), TIMEOUT, folds=CV_FOLDS)
    months = lambda folds: np.isin(tra.month.values, [m for f in folds for m in f])
    fit_m, judge_m = months(FOLDS), months(HELD)
    ok = np.isfinite(p5) & np.isfinite(base)
    assert ok.all(), f"{(~ok).sum():,} rows lack a prediction; LT2 refuses to judge a partial vector"

    _r, w, _, _ = blended(tra, p5, base, fit_m)
    r_fit, _, _fb, f_base = blended(tra, p5, base, fit_m, w=w)
    r_judge, _, e_blend, e_base = blended(tra, p5, base, judge_m, w=w)
    v1 = paired_delta(e_base, e_blend, tra.day.values[judge_m])
    passed = v1.delta >= GATE and v1.lo > 0

    read2 = ["Read 2 was NOT run: read 1 did not clear the floor, so the untouched holdout was never met and this lane "
             "closes at zero holdout fits."]
    if passed:
        y, sb3 = tea.y.values, sb3_predict(tra, tea)
        lvl = floored(fit(win, tra, tea) + tea.z_aobt.values)
        v2 = paired_delta(sb3 - y, floored(w * lvl + (1 - w) * sb3) - y, tea.day.values)
        read2 = [f"Holdout, LT2 alone {rmse(lvl - y):.2f} s, blended onto SB3 at the SAME weight {w:.4f} fitted out of "
                 f"fold on the search months: " + verdict_line("LT2+SB3", "SB3", v2, k=K)]

    txt = "\n".join([
        "# LT2: a tuned piecewise-linear tree, searched on the blend", "",
        f"Regenerate with `python -m taxi.lt2`. Declared in `docs/EXPERIMENT-PROTOCOL.md` section 4 before this module "
        f"existed. {achieved(next(x for x in SEARCHES if x.module == "taxi.lt2"))} completed trials of a declared "
        f"{K}, TPE seed 0, {(time.time() - t0) / 3600:.1f} h of a {TIMEOUT / 3600:.0f} h clock.", "",
        "## The winning cell", "",
        "| axis | value |", "|---|---|",
        *[f"| {k} | {v} |" for k, v in sorted(win.items())], "",
        "## Read 1: the months no cell was ever ranked on", "",
        f"The weight is fitted on the {int(fit_m.sum()):,} rows of the search months and never re-fitted. The verdict is "
        f"read on the {int(judge_m.sum()):,} rows of months 4, 5, 9 and 10, which no cell in this campaign was chosen on, "
        f"so the cell and the weight are both fixed by rows disjoint from every judged row and the optimism bound is 0.0 s.",
        "",
        "| rows | MC2 out of fold | blend | gain |", "|---|---|---|---|",
        f"| search months, {int(fit_m.sum()):,} | {rmse(f_base):.2f} | {r_fit:.2f} | {rmse(f_base) - r_fit:+.2f} |",
        f"| held months, {int(judge_m.sum()):,} | {rmse(e_base):.2f} | {r_judge:.2f} | {rmse(e_base) - r_judge:+.2f} |",
        "",
        f"The gap between those two gains is the SELECTION advantage this cell bought from the {K} trials, and it is "
        f"measured here rather than bounded.", "",
        f"**{'PASS' if passed else 'FAIL'}**: {v1.delta:+.2f} s [{v1.lo:+.2f}, {v1.hi:+.2f}] against a {GATE:.0f} s floor "
        f"and a 0.0 s bound, weight {w:.4f}.", "",
        "## Read 2: the untouched holdout", "", *read2, ""])
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))
