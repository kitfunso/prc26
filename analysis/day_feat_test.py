"""Does an airport-day/hour visible-stamp signature help the lane-A stamp classifier?
Baseline cols = CAT+NUM+DAY, test cols = CAT+NUM+DAY+NEWDAY. CPU LightGBM, small (num_leaves=63, lr=0.05).
Physical (class 3) rows subsampled to PHYS_CAP per training fold for CPU time; noted in the report.
`PYTHONPATH=/c/Users/skf_s/prc26 python analysis/day_feat_test.py`"""
from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from data import CACHE, rmse
from features import CAT, DAY, NUM, build
from mix import Z, label
from train import FOLDS, HOLDOUT

PHYS_CAP = 200_000
CLF = dict(objective="multiclass", num_class=4, learning_rate=0.05, num_leaves=63, min_child_samples=50,
           feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=3.0, n_estimators=2000,
           verbose=-1, random_state=0, n_jobs=24)
REG = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_child_samples=50, feature_fraction=0.8,
           bagging_fraction=0.8, bagging_freq=1, lambda_l2=3.0, n_estimators=2000, verbose=-1, random_state=0, n_jobs=24)
NEWDAY = ["day_aobt0_shr", "day_aobtm5_shr", "day_eobtm5_shr", "day_dae0_shr", "day_dle0_shr",
          "h_aobt0_shr", "h_aobtm5_shr", "h_eobtm5_shr", "h_dae0_shr", "h_dle0_shr"]


def newday_features(f: pd.DataFrame) -> pd.DataFrame:
    """Airport-day and +-2h-window shares of visible AOBT/EOBT/LOBT patterns (no y/label use: safe as a classifier input)."""
    d = f.assign(dle0=(f["z_eobt"] - f["z_lobt"] == 0).astype(int), aobt0=(f["aobt_sec"] == 0).astype(int),
                 dae0=(f["d_ae"] == 0).astype(int))
    g = d.groupby(["split", "apt", "day"], observed=True)
    day = pd.DataFrame({"day_aobt0_shr": g["aobt0"].transform("mean"), "day_aobtm5_shr": g["aobt_m5"].transform("mean"),
                        "day_eobtm5_shr": g["eobt_m5"].transform("mean"), "day_dae0_shr": g["dae0"].transform("mean"),
                        "day_dle0_shr": g["dle0"].transform("mean")}, index=d.index)

    hg = d.groupby(["split", "apt", "day", "hour"], observed=True)
    cnt = hg.size().rename("n")
    sums = hg[["aobt0", "aobt_m5", "eobt_m5", "dae0", "dle0"]].sum()
    tbl = pd.concat([cnt, sums], axis=1).reset_index()

    def window(g: pd.DataFrame) -> pd.DataFrame:
        g = g.set_index("hour").reindex(range(24), fill_value=0)
        w = g.rolling(5, center=True, min_periods=1).sum()
        return w

    win = tbl.groupby(["split", "apt", "day"], group_keys=True, observed=True).apply(window, include_groups=False)
    win = win.reset_index()
    n = win["n"].replace(0, np.nan)
    for c, out in (("aobt0", "h_aobt0_shr"), ("aobt_m5", "h_aobtm5_shr"), ("eobt_m5", "h_eobtm5_shr"),
                   ("dae0", "h_dae0_shr"), ("dle0", "h_dle0_shr")):
        win[out] = (win[c] / n).fillna(0)
    win = win[["split", "apt", "day", "hour", "h_aobt0_shr", "h_aobtm5_shr", "h_eobtm5_shr", "h_dae0_shr", "h_dle0_shr"]]
    day = day.join(d[["split", "apt", "day", "hour"]])
    day = day.merge(win, on=["split", "apt", "day", "hour"], how="left")
    day.index = d.index
    return day.drop(columns=["split", "apt", "day", "hour"])


def cap_phys(tr: pd.DataFrame, lab: np.ndarray, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    phys = np.where(lab == 3)[0]
    if len(phys) <= PHYS_CAP:
        return tr, lab
    rng = np.random.default_rng(seed)
    drop = rng.choice(phys, len(phys) - PHYS_CAP, replace=False)
    keep = np.ones(len(tr), dtype=bool); keep[drop] = False
    return tr[keep], lab[keep]


def yhat_from(P: np.ndarray, te: pd.DataFrame, mu: np.ndarray) -> np.ndarray:
    return sum(P[:, k] * np.nan_to_num(te[z].values) for k, z in enumerate(Z)) + P[:, 3] * mu


def run_variant(cols: list[str], tr: pd.DataFrame, lab_tr: np.ndarray, va: pd.DataFrame, lab_va: np.ndarray,
                 mu_va: np.ndarray) -> tuple[float, float]:
    trc, labc = cap_phys(tr, lab_tr, seed=0)
    m = lgb.LGBMClassifier(**CLF)
    m.fit(trc[cols], labc, eval_set=[(va[cols], lab_va)], categorical_feature=CAT,
          callbacks=[lgb.early_stopping(100, verbose=False)])
    P = m.predict_proba(va[cols])
    for k, z in enumerate(Z):
        P[va[z].isna().values, k] = 0
    P /= P.sum(1, keepdims=True)
    ll = -np.log(np.clip(P[np.arange(len(va)), lab_va], 1e-12, 1)).mean()
    yhat = yhat_from(P, va, mu_va)
    return ll, rmse(va["y"].values, yhat)


def fit_mu(tr: pd.DataFrame, lab_tr: np.ndarray, va: pd.DataFrame) -> np.ndarray:
    phys = tr[lab_tr == 3]
    if len(phys) > PHYS_CAP:
        phys = phys.sample(PHYS_CAP, random_state=0)
    m = lgb.LGBMRegressor(**REG)
    m.fit(phys[CAT + NUM + DAY], phys["y"], categorical_feature=CAT)
    return m.predict(va[CAT + NUM + DAY])


def main() -> None:
    f = build()
    f = pd.concat([f, newday_features(f)], axis=1)
    tr_all = f[(f["split"] == "train") & ~f["month"].isin(HOLDOUT) & (f["has_plan"] == 1)]
    ho = f[(f["split"] == "train") & f["month"].isin(HOLDOUT) & (f["has_plan"] == 1)]

    base_cols, new_cols = CAT + NUM + DAY, CAT + NUM + DAY + NEWDAY
    rows, t0 = [], time.time()
    for i, months in enumerate(FOLDS):
        va_m = tr_all["month"].isin(months).values
        tr, va = tr_all[~va_m], tr_all[va_m]
        lab_tr, lab_va = label(tr), label(va)
        mu_va = fit_mu(tr, lab_tr, va)
        ll_b, rmse_b = run_variant(base_cols, tr, lab_tr, va, lab_va, mu_va)
        ll_n, rmse_n = run_variant(new_cols, tr, lab_tr, va, lab_va, mu_va)
        rows.append((f"fold{i}", months, len(va), ll_b, ll_n, rmse_b, rmse_n))
        print(f"fold {i} {months}: n={len(va)} logloss base={ll_b:.4f} new={ll_n:.4f} | rmse base={rmse_b:.2f} new={rmse_n:.2f}  ({time.time()-t0:.0f}s)", flush=True)

    lab_tr, lab_ho = label(tr_all), label(ho)
    mu_ho = fit_mu(tr_all, lab_tr, ho)
    ll_b, rmse_b = run_variant(base_cols, tr_all, lab_tr, ho, lab_ho, mu_ho)
    ll_n, rmse_n = run_variant(new_cols, tr_all, lab_tr, ho, lab_ho, mu_ho)
    rows.append(("holdout", HOLDOUT, len(ho), ll_b, ll_n, rmse_b, rmse_n))
    print(f"holdout: n={len(ho)} logloss base={ll_b:.4f} new={ll_n:.4f} | rmse base={rmse_b:.2f} new={rmse_n:.2f}  ({time.time()-t0:.0f}s)")

    out = pd.DataFrame(rows, columns=["fold", "months", "n", "logloss_base", "logloss_new", "rmse_base", "rmse_new"])
    out.to_csv(CACHE.parent / "analysis_day_feat_test.csv", index=False)
    print("\n", out.to_string(index=False))


if __name__ == "__main__":
    main()
