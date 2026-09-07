"""AV: adversarial validation of the 2025 holdout against the 2026 ranking rows. Diagnostic, never a verdict (protocol section 2).

HB priced the row-mix shift on an airport-and-month partition and found +1.1512 s of lane A. That partition has 20 cells; this one
is the whole 52-column feature space, so it can see a shift HB's cells average away, and the importance-weighted holdout RMSE is
the standard estimator of what the board will score.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score

from taxi.champ import COLS_D2, lane_a
from taxi.data import features
from taxi.l2 import floored
from taxi.sb3 import predict as sb3_predict
from taxi.verdict import rmse
from taxi.weather import join, load_metar

RANKING = "data/ranking.parquet"
CAT = ["apt", "runway", "stand", "actype", "wake", "segment", "operator", "ftype", "frule", "ades"]
KW = dict(iterations=400, depth=4, learning_rate=0.1, l2_leaf_reg=3.0, random_seed=0, verbose=False, allow_writing_files=False)
CLIP = 0.99  # weights are a ratio of densities and the tail is where the estimate is least trustworthy; both numbers are reported


def _pool(d: pd.DataFrame, y: np.ndarray | None = None, cols: list[str] | None = None) -> Pool:
    cs = COLS_D2 if cols is None else cols
    x = d[cs].copy()
    cat = [c for c in CAT if c in cs]
    for c in cat:
        x[c] = x[c].astype(str)
    return Pool(x, y, cat_features=cat)


def _wrmse(err: np.ndarray, w: np.ndarray) -> float:
    return float(np.sqrt(np.average(err ** 2, weights=w)))


def _shift_ci(err: np.ndarray, w: np.ndarray, days: np.ndarray, reps: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Day-block bootstrap of the weighted minus unweighted RMSE, both computed on the same resample so the difference is paired."""
    _, grp = np.unique(days, return_inverse=True)
    nb = int(grp.max()) + 1
    n, sse = np.bincount(grp, minlength=nb).astype(float), np.bincount(grp, weights=err ** 2, minlength=nb)
    sw, swe = np.bincount(grp, weights=w, minlength=nb), np.bincount(grp, weights=w * err ** 2, minlength=nb)
    pick = np.random.default_rng(seed).integers(0, nb, size=(reps, nb))
    d = np.sqrt(swe[pick].sum(axis=1) / sw[pick].sum(axis=1)) - np.sqrt(sse[pick].sum(axis=1) / n[pick].sum(axis=1))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> None:
    metar = load_metar()
    tra, tea = lane_a()
    rk = join(features([RANKING]), metar)
    rka = rk[rk.has_fpl].reset_index(drop=True)
    rng = np.random.default_rng(0)
    half = rng.permutation(len(tea)) < len(tea) // 2  # the classifier is fit on one half and scores the other, so no row is scored in sample
    src, dst = tea, rka.iloc[rng.permutation(len(rka))[:len(tea)]].reset_index(drop=True)
    dhalf = rng.permutation(len(dst)) < len(dst) // 2
    fit = pd.concat([src[half], dst[dhalf]], ignore_index=True)
    lab = np.r_[np.zeros(int(half.sum())), np.ones(int(dhalf.sum()))]
    m = CatBoostClassifier(**KW).fit(_pool(fit, lab))
    ps, pd_ = m.predict_proba(_pool(src[~half]))[:, 1], m.predict_proba(_pool(dst[~dhalf]))[:, 1]
    auc = roc_auc_score(np.r_[np.zeros(len(ps)), np.ones(len(pd_))], np.r_[ps, pd_])
    imp = sorted(zip(m.feature_names_, m.get_feature_importance()), key=lambda t: -t[1])[:8]
    dry = [c for c in COLS_D2 if not c.startswith("w_")]  # the same question with the weather realisation removed: is the OPERATION different?
    md = CatBoostClassifier(**KW).fit(_pool(fit, lab, dry))
    auc_dry = roc_auc_score(np.r_[np.zeros(int((~half).sum())), np.ones(int((~dhalf).sum()))],
                            np.r_[md.predict_proba(_pool(src[~half], cols=dry))[:, 1], md.predict_proba(_pool(dst[~dhalf], cols=dry))[:, 1]])

    err = (floored(sb3_predict(tra, tea)) - tea.y.values)[~half]  # the champion's own errors on the rows the classifier never saw
    w = ps / np.clip(1 - ps, 1e-6, None)
    cap = float(np.quantile(w, CLIP))
    wc = np.clip(w, None, cap)
    ess = float(w.sum() ** 2 / (w ** 2).sum())
    day = tea.day.values[~half]
    lo, hi = _shift_ci(err, w, day)
    loc, hic = _shift_ci(err, wc, day)
    rows = [f"| unweighted (the number every verdict rides) | {rmse(err):.4f} | {len(err):,} | | |",
            f"| importance weighted, raw | {_wrmse(err, w):.4f} | {ess:,.0f} | {_wrmse(err, w) - rmse(err):+.4f} | [{lo:+.2f}, {hi:+.2f}] |",
            f"| importance weighted, capped at the {CLIP:.0%} weight ({cap:.2f}) | {_wrmse(err, wc):.4f} | "
            f"{wc.sum() ** 2 / (wc ** 2).sum():,.0f} | {_wrmse(err, wc) - rmse(err):+.4f} | [{loc:+.2f}, {hic:+.2f}] |"]

    out = ["# AV: does the 2026 row set sit where the 2025 holdout sits?", "",
           "Diagnostic, never a verdict, no cell declared. Regenerate with `python -m taxi.av`. A CatBoost classifier separates the "
           f"{len(src):,} lane-A holdout rows from an equal sample of the {len(rka):,} lane-A ranking rows on the champion's own "
           f"{len(COLS_D2)} columns, fit on half of each side and scored on the other half. Both sides are January and July, one year apart, so "
           "the calendar is held and only the year moves. If the two row sets were exchangeable the classifier would read 0.5.", "",
           f"## Separability", "", f"Out-of-sample AUC **{auc:.4f}** on {len(ps) + len(pd_):,} scored rows, and **{auc_dry:.4f}** on the same rows "
           f"with the {len(COLS_D2) - len(dry)} weather columns removed, which separates a different weather realisation from a different "
           f"operation.", "",
           "| column | importance |", "|---|---|", *[f"| {c} | {v:.2f} |" for c, v in imp], "",
           "## What the shift is worth on the champion's own errors", "",
           "The importance weight `p / (1 - p)` turns the holdout into an unbiased sample of the ranking set under covariate shift, so the "
           "weighted RMSE below is the estimate of what lane A scores on the board, against HB's airport-and-month figure of 233.3014 s.", "",
           "| weighting | lane-A RMSE (s) | effective rows | shift | 95% day-block interval |", "|---|---|---|---|---|", *rows, ""]
    text = "\n".join(out)
    with open("docs/AV.md", "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
