"""Does the arrival recorder-mode signal help the lane-B copy/near/phys classifier? Lane B rows, train split
only. classes() imported from laneb.py (frozen name; that file is being edited elsewhere so nothing else is
imported from it - CLF_LGB and the cross/sched_hour derivation are copied here instead).
`PYTHONPATH=/c/Users/skf_s/prc26 python analysis/arr_feat_test_b.py`"""
from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from data import CACHE
from features import build
from laneb import classes
from mix import B_CAT, B_NUM
from train import FOLDS, HOLDOUT
from arrmode import build as build_arr
from features import ARR

CLF_LGB = dict(objective="multiclass", learning_rate=0.05, num_leaves=63, min_child_samples=30, feature_fraction=0.8,
               bagging_fraction=0.8, bagging_freq=1, lambda_l2=3.0, n_estimators=2000, verbose=-1, random_state=0, n_jobs=8)
BASE_COLS = B_CAT + B_NUM + ["cross", "sched_hour"]


def augment(f: pd.DataFrame) -> pd.DataFrame:
    """Cross-midnight flag and schedule hour, copied from laneb.augment (laneb.py is being edited elsewhere)."""
    sched = f["t_off"].values - f["z_sched"].values
    return f.assign(cross=((sched // 86400) < (f["t_off"].values // 86400)).astype(int),
                     sched_hour=((sched % 86400) // 3600).astype(float))


def fit_clf(cols: list[str], tr: pd.DataFrame, lab: np.ndarray, keep: np.ndarray, va: pd.DataFrame,
            vlab: np.ndarray, vkeep: np.ndarray) -> np.ndarray:
    """Multiclass over whichever of {copy, near, phys} are present in tr, expanded back to a fixed 3-column P."""
    present = sorted(set(lab[keep].tolist()))
    remap = {v: i for i, v in enumerate(present)}
    y = np.vectorize(remap.get)(lab[keep])
    params = {**CLF_LGB, "num_class": len(present)}
    n = params.pop("n_estimators")
    m = lgb.LGBMClassifier(**{**params, "n_estimators": n})
    yv = np.vectorize(remap.get)(vlab[vkeep])
    m.fit(tr[cols][keep], y, eval_set=[(va[cols][vkeep], yv)], categorical_feature=B_CAT,
          callbacks=[lgb.early_stopping(100, verbose=False)])
    raw = m.predict_proba(va[cols][vkeep])
    P = np.zeros((int(vkeep.sum()), 3))
    for v, i in remap.items():
        P[:, v] = raw[:, i]
    return P


def score(cols: list[str], tr: pd.DataFrame, va: pd.DataFrame) -> tuple[float, np.ndarray, np.ndarray]:
    copy, near, phys, shift = classes(tr)
    lab, keep = np.select([copy, near], [0, 1], default=2), ~shift
    vcopy, vnear, vphys, vshift = classes(va)
    vlab, vkeep = np.select([vcopy, vnear], [0, 1], default=2), ~vshift
    P = fit_clf(cols, tr, lab, keep, va, vlab, vkeep)
    p = np.clip(P, 1e-9, 1); p /= p.sum(1, keepdims=True)
    ll = log_loss(vlab[vkeep], p, labels=[0, 1, 2])
    return ll, p, vlab[vkeep]


def main() -> None:
    f = augment(build().merge(build_arr(), on="id", how="left"))
    b = f[f["has_plan"] == 0]
    tr_all = b[(b["split"] == "train") & ~b["month"].isin(HOLDOUT)]
    ho = b[(b["split"] == "train") & b["month"].isin(HOLDOUT)]

    t0 = time.time()
    ll_sum = {"base": 0.0, "arr": 0.0}; n_sum = 0
    for i, months in enumerate(FOLDS):
        va_m = tr_all["month"].isin(months).values
        trn, van = tr_all[~va_m], tr_all[va_m]
        ll_b, _, _ = score(BASE_COLS, trn, van)
        ll_a, _, _ = score(BASE_COLS + ARR, trn, van)
        _, _, vlab_keep = score(BASE_COLS, trn, van)  # cheap: reuse keep-mask size for weighting
        n = len(vlab_keep)
        ll_sum["base"] += ll_b * n; ll_sum["arr"] += ll_a * n; n_sum += n
        print(f"fold {i} {months}: n={n} logloss base={ll_b:.4f} arr={ll_a:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"CV logloss (n-weighted): base={ll_sum['base']/n_sum:.4f} arr={ll_sum['arr']/n_sum:.4f}")

    ll_ho_b, p_b, vlab_b = score(BASE_COLS, tr_all, ho)
    ll_ho_a, p_a, vlab_a = score(BASE_COLS + ARR, tr_all, ho)
    brier_b = float(np.mean((p_b[:, 0] - (vlab_b == 0)) ** 2))
    brier_a = float(np.mean((p_a[:, 0] - (vlab_a == 0)) ** 2))
    pcopy_b = float(p_b[vlab_b == 0, 0].mean())
    pcopy_a = float(p_a[vlab_a == 0, 0].mean())
    print(f"holdout logloss: base={ll_ho_b:.4f} arr={ll_ho_a:.4f}")
    print(f"holdout copy-class Brier: base={brier_b:.4f} arr={brier_a:.4f}")
    print(f"holdout mean P(copy) on true-copy rows: base={pcopy_b:.4f} arr={pcopy_a:.4f}, n_copy={int((vlab_b==0).sum())}")

    pd.DataFrame([{"metric": "cv_logloss", "base": ll_sum["base"]/n_sum, "arr": ll_sum["arr"]/n_sum},
                  {"metric": "holdout_logloss", "base": ll_ho_b, "arr": ll_ho_a},
                  {"metric": "holdout_copy_brier", "base": brier_b, "arr": brier_a},
                  {"metric": "holdout_mean_p_copy_on_true_copy", "base": pcopy_b, "arr": pcopy_a}]
                 ).to_csv(CACHE.parent / "analysis_arr_feat_test_b.csv", index=False)


if __name__ == "__main__":
    main()
