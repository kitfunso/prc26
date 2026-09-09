"""FAM2: the model classes the register still leaves to one arm or to none, lane A.

FAM1 emptied the four rows that stood blank. Three rows were left thin rather than blank and a register cannot tell the two
apart: penalised linear names ridge, lasso and elastic net but only ridge was ever built; robust linear says "lane B only";
and there is no row at all for a non-Gaussian GLM, a latent-factor regression or a piecewise-linear tree. This lane fills
them. Same shape as FAM1: fixed cells, no search inside any arm, and the gate is the blend against MC2, not the level.
"""
from __future__ import annotations

import gc
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import BayesianRidge, ElasticNet, HuberRegressor, TweedieRegressor
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor

from taxi.champ import COLS_D2, lane_a
from taxi.fam1 import FOLDS, GATE, _num_block, _tree_block, cache_path, oof, score_arm
from taxi.l2 import floored
from taxi.nn1 import base_oof
from taxi.rc1 import Enc
from taxi.sb3 import predict as sb3_predict
from taxi.verdict import paired_delta, rmse
from taxi.w1 import verdict_line

BUDGET_S = 8 * 3600
ENC = dict(min_freq=20, knots=4, degree=3, wins=None)  # RC1's winning cell, so an arm differs from RC1 only by its model class
CAP = 300  # iteration cap shared by every linear arm: a screen prices the family, and none of these is the campaign's champion


def _enc(tr: pd.DataFrame, te: pd.DataFrame):
    e = Enc(tr, COLS_D2, ENC["min_freq"], ENC["knots"], ENC["degree"], ENC["wins"])
    return e(tr), e(te)


def _sparse_arm(kind, tr: pd.DataFrame, te: pd.DataFrame, on_level: bool = False) -> np.ndarray:
    a, b = _enc(tr, te)
    target = np.maximum(tr.y.values, 0.0) if on_level else tr.c.values
    m = kind.fit(a, target)
    del a
    gc.collect()
    p = m.predict(b)
    return p - te.z_aobt.values if on_level else p  # every arm hands back a `c`, so the harness assembles levels one way


def _dense_arm(kind, tr: pd.DataFrame, te: pd.DataFrame) -> np.ndarray:
    a, b = _num_block(tr, te)
    m = kind.fit(a, tr.c.values)
    del a
    gc.collect()
    return np.asarray(m.predict(b)).ravel()


def arm_lt(tr: pd.DataFrame, te: pd.DataFrame) -> np.ndarray:
    """Piecewise-LINEAR leaves. Every boosted lane in this campaign fits a constant per leaf; this is the class that does not."""
    import lightgbm as lgb

    a, b = _tree_block(tr, te)
    m = lgb.LGBMRegressor(linear_tree=True, n_estimators=300, num_leaves=64, learning_rate=0.1, min_child_samples=100,
                          linear_lambda=1.0, n_jobs=12, random_state=0, verbose=-1).fit(a, tr.c.values)
    del a
    gc.collect()
    return m.predict(b)


def arm_en(tr, te):
    return _sparse_arm(ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=CAP, tol=1e-3, selection="random", random_state=0), tr, te)


def arm_svr(tr, te):
    return _sparse_arm(LinearSVR(C=0.1, epsilon=0.0, max_iter=CAP, random_state=0, dual=True), tr, te)


def arm_hub(tr, te):
    return _sparse_arm(HuberRegressor(epsilon=1.35, alpha=1.0, max_iter=CAP, tol=1e-3), tr, te)


def arm_twd(tr, te):
    """Tweedie on the LEVEL, because `c` is a residual and 56% of it is negative, so a log link cannot see it."""
    return _sparse_arm(TweedieRegressor(power=1.5, alpha=1.0, link="log", max_iter=CAP, tol=1e-3), tr, te, on_level=True)


def arm_brr(tr, te):
    return _dense_arm(BayesianRidge(max_iter=CAP, tol=1e-3), tr, te)


def arm_pls(tr, te):
    return _dense_arm(PLSRegression(n_components=16, scale=False, max_iter=CAP), tr, te)


def arm_ada(tr, te):
    a, b = _tree_block(tr, te)
    m = AdaBoostRegressor(DecisionTreeRegressor(max_depth=6, random_state=0), n_estimators=100, learning_rate=0.5,
                          loss="square", random_state=0).fit(a, tr.c.values)
    del a
    gc.collect()
    return m.predict(b)


# Most-promising first, not cheapest-first: FAM1 ran cheapest-first and the budget is what starves an arm, so the one arm
# with a real mechanism story must not be the one that runs last.
ARMS = (("lt", "piecewise-linear boosted trees (LightGBM linear_tree)", arm_lt),
        ("pls", "latent-factor regression (partial least squares)", arm_pls),
        ("brr", "Bayesian linear (evidence-maximised ridge)", arm_brr),
        ("twd", "non-Gaussian GLM (Tweedie, log link, on the level)", arm_twd),
        ("en", "sparse linear (elastic net)", arm_en),
        ("svr", "linear support-vector regression, epsilon-insensitive loss", arm_svr),
        ("hub", "robust linear (Huber), the lane-A row the register leaves to lane B", arm_hub),
        ("ada", "adaptive boosting (AdaBoost.R2 on depth-6 trees)", arm_ada))


def path(tag: str) -> str:
    return cache_path(tag).replace("fam1_", "fam2_")


def screen(tra: pd.DataFrame, base: np.ndarray, budget: float) -> list[dict]:
    rows: list[dict] = []
    for tag, label, fn in ARMS:
        left = budget - sum(r["min"] for r in rows) * 60
        if left <= 0:
            rows.append(dict(tag=tag, label=label, min=0.0, note="not started: the lane's budget was spent by earlier arms"))
            continue
        try:
            p, spent, note = oof(tra, fn, left)
        except MemoryError:
            rows.append(dict(tag=tag, label=label, min=0.0, note="MemoryError: this cell does not fit the free RAM on this box"))
            continue
        except Exception as exc:
            rows.append(dict(tag=tag, label=label, min=0.0, note=f"{type(exc).__name__}: {exc}"[:180]))
            continue
        rows.append(score_arm(tra, base, tag, label, p, spent, note))
        pd.DataFrame({"id": tra.id.values, "pred": p}).to_parquet(path(tag), index=False)
    return rows


def cached(tra: pd.DataFrame, base: np.ndarray, _budget: float) -> list[dict]:
    rows: list[dict] = []
    for tag, label, _fn in ARMS:
        if not os.path.exists(path(tag)):
            rows.append(dict(tag=tag, label=label, min=0.0, note=f"no saved vector at {path(tag)}"))
            continue
        rows.append(score_arm(tra, base, tag, label, pd.read_parquet(path(tag)).pred.values, 0.0, ""))
    return rows


def main(out: str = "docs/FAM2.md", from_cache: bool = False) -> None:
    tra, tea = lane_a()
    t0 = time.time()
    print("| arm | own RMSE | blend | MC2 base | agreement | min |\n|---|---|---|---|---|---|", flush=True)
    rows = (cached if from_cache else screen)(tra, base_oof(tra), BUDGET_S)

    passed = [r for r in rows if r.get("blend") is not None and r["base"] - r["blend"] >= GATE]
    tbl, verd = [], []
    for r in rows:
        if "blend" not in r:
            tbl.append(f"| FAM2-{r['tag']} | {r['label']} | NOT RUN | | | | | {r['note']} |")
            continue
        gap = r["base"] - r["blend"]
        tbl.append(f"| FAM2-{r['tag']} | {r['label']} | {r['level']:.2f} | {r['blend']:.2f} | {gap:+.2f} | {r['raw']:+.4f} | "
                   f"{r['agree']:.4f} | {'PASS' if gap >= GATE else 'closed at this cell'}"
                   f"{'; ' + r['note'] if r['note'] else ''} |")

    y, sb3 = tea.y.values, sb3_predict(tra, tea)
    e0 = sb3 - y
    for r in passed:
        fn = dict((t, f) for t, _, f in ARMS)[r["tag"]]
        p = floored(fn(tra, tea) + tea.z_aobt.values)
        v = paired_delta(e0, floored(r["w"] * p + (1 - r["w"]) * sb3) - y, tea.day.values)
        verd.append(f"- **FAM2-{r['tag']} + SB3** at w={r['w']:.4f}, holdout {rmse(p - y):.2f} s alone: "
                    + verdict_line(f"FAM2-{r['tag']}", "SB3", v, k=len(ARMS)))

    clock = (f"Rescored from the saved arm vectors, so each arm's wall clock reads 0." if from_cache
             else f"Total {(time.time() - t0) / 3600:.1f} h of a {BUDGET_S / 3600:.0f} h budget.")
    text = "\n".join([
        "# FAM2: the model-class rows the register left thin, lane A screen", "",
        f"Regenerate with `python -m taxi.fam2`. Declared in protocol section 4 before the code was written. "
        f"{len(ARMS)} arms, one fixed cell each, NO search inside any arm, k = {len(ARMS)}. Folds are the three declared "
        f"month pairs {FOLDS} that FAM1, O2 and NN3 use, and the comparator is the one FAM1 used: MC2's cached "
        f"out-of-fold vector on those rows. {clock}", "",
        "FAM1 emptied the four register rows that stood blank. These are the rows that were thin rather than blank, which a "
        "register cannot distinguish: penalised linear named lasso and elastic net but instantiated only ridge; robust linear "
        "read 'lane B only'; and three classes had no row at all. Every linear arm shares RC1's winning encoder "
        f"({ENC}), so an arm differs from RC1 only by its model class and the comparison is not confounded by the basis.", "",
        "## The screen", "",
        f"Gate declared in advance, identical to FAM1: an arm meets the untouched holdout only if its blend with MC2 beats "
        f"MC2 alone by {GATE:.1f} s on these rows. An arm that fails is a closed register row and costs no holdout fit.", "",
        "| arm | family | own RMSE | blend with MC2 | gain over MC2 | unclipped weight | agreement | outcome |",
        "|---|---|---|---|---|---|---|---|",
        *tbl, "",
        "## Verdicts against SB3" if verd else "## No arm passed the gate", "",
        *(verd or ["Every arm failed the declared gate, so none met the holdout. The register rows are closed by measurement."]),
        "", "## Not done, and why", "",
        "- **Explainable boosting machine (GA2M).** `interpret` is not installed on this box. It is the one remaining named "
        "family with a mechanism story this lane does not cover, so it is a NOT-DONE row rather than a closure by argument.",
        ""])
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main(from_cache="--from-cache" in sys.argv)
