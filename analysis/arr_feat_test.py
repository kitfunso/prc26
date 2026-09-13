"""Does the arrival recorder-mode signal (analysis/arrmode.py) help the lane-A stamp classifier, on top of
the airport-day/hour visible-stamp signature already tested in day_feat_test.py? Same harness: CAT+NUM+DAY
baseline, CPU LightGBM, physical class capped at PHYS_CAP rows per fold.
`PYTHONPATH=/c/Users/skf_s/prc26 python analysis/arr_feat_test.py`"""
from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from data import CACHE, rmse
from day_feat_test import CLF, NEWDAY, cap_phys, fit_mu, newday_features
from features import CAT, DAY, NUM, build
from mix import Z, label
from train import FOLDS, HOLDOUT
from arrmode import build as build_arr
from features import ARR

VARIANTS = {"base": [], "newday": NEWDAY, "arr": ARR, "both": NEWDAY + ARR}


def score(cols: list[str], tr: pd.DataFrame, lab_tr: np.ndarray, va: pd.DataFrame, lab_va: np.ndarray,
          mu_va: np.ndarray) -> dict[str, float]:
    """Fit the multiclass stamp classifier on `cols`, report log loss, mixture RMSE, class recall, SSE share."""
    trc, labc = cap_phys(tr, lab_tr, seed=0)
    m = lgb.LGBMClassifier(**CLF)
    m.fit(trc[cols], labc, eval_set=[(va[cols], lab_va)], categorical_feature=CAT,
          callbacks=[lgb.early_stopping(100, verbose=False)])
    P = m.predict_proba(va[cols])
    for k, z in enumerate(Z):
        P[va[z].isna().values, k] = 0
    P /= P.sum(1, keepdims=True)
    ll = -np.log(np.clip(P[np.arange(len(va)), lab_va], 1e-12, 1)).mean()
    yhat = sum(P[:, k] * np.nan_to_num(va[z].values) for k, z in enumerate(Z)) + P[:, 3] * mu_va
    pred = P.argmax(1)
    recall_eobt = float((pred[lab_va == 1] == 1).mean()) if (lab_va == 1).any() else np.nan
    recall_sched = float((pred[lab_va == 2] == 2).mean()) if (lab_va == 2).any() else np.nan
    cell = (lab_va == 1) & (pred == 3)  # true eobt predicted other: the mixture's main blind spot
    err2 = (va["y"].values - yhat) ** 2
    sse_share = float(err2[cell].sum() / err2.sum()) if err2.sum() > 0 else np.nan
    return dict(logloss=ll, rmse=rmse(va["y"].values, yhat), recall_eobt=recall_eobt,
                recall_sched=recall_sched, sse_share_eobt_other=sse_share)


def main() -> None:
    f = build()
    f = pd.concat([f, newday_features(f)], axis=1).merge(build_arr(), on="id", how="left")
    tr_all = f[(f["split"] == "train") & ~f["month"].isin(HOLDOUT) & (f["has_plan"] == 1)]
    ho = f[(f["split"] == "train") & f["month"].isin(HOLDOUT) & (f["has_plan"] == 1)]
    base_cols = CAT + NUM + DAY

    rows, t0 = [], time.time()
    for i, months in enumerate(FOLDS):
        va_m = tr_all["month"].isin(months).values
        tr, va = tr_all[~va_m], tr_all[va_m]
        lab_tr, lab_va = label(tr), label(va)
        mu_va = fit_mu(tr, lab_tr, va)
        for vname, extra in VARIANTS.items():
            m = score(base_cols + extra, tr, lab_tr, va, lab_va, mu_va)
            rows.append({"fold": f"fold{i}", "variant": vname, "n": len(va), **m})
        print(f"fold {i} {months}: " + " | ".join(f"{v}: rmse={r['rmse']:.2f} ll={r['logloss']:.4f}"
              for v, r in zip(VARIANTS, rows[-len(VARIANTS):])) + f"  ({time.time()-t0:.0f}s)", flush=True)

    lab_tr, lab_ho = label(tr_all), label(ho)
    mu_ho = fit_mu(tr_all, lab_tr, ho)
    for vname, extra in VARIANTS.items():
        m = score(base_cols + extra, tr_all, lab_tr, ho, lab_ho, mu_ho)
        rows.append({"fold": "holdout", "variant": vname, "n": len(ho), **m})
    print("holdout: " + " | ".join(f"{v}: rmse={r['rmse']:.2f} ll={r['logloss']:.4f}"
          for v, r in zip(VARIANTS, rows[-len(VARIANTS):])) + f"  ({time.time()-t0:.0f}s)")

    out = pd.DataFrame(rows)
    out.to_csv(CACHE.parent / "analysis_arr_feat_test.csv", index=False)
    print("\n", out.round(4).to_string(index=False))

    ho_rows = out[out["fold"] == "holdout"].set_index("variant")
    base_rmse = ho_rows.loc["base", "rmse"]
    fold_rows = out[out["fold"] != "holdout"]
    for v in ("arr", "both"):
        beats_base_every_fold = all(fold_rows[fold_rows["variant"] == v]["rmse"].values <
                                     fold_rows[fold_rows["variant"] == "base"]["rmse"].values)
        delta = base_rmse - ho_rows.loc[v, "rmse"]
        passed = delta > 2.16 and beats_base_every_fold
        print(f"Test A verdict [{v}]: holdout delta={delta:+.2f} (need >2.16), "
              f"beats base every fold={beats_base_every_fold} -> {'PASS' if passed else 'FAIL'}")


if __name__ == "__main__":
    main()
