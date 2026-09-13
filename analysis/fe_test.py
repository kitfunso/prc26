"""Do the fe.py feature groups help the lane-A stamp classifier on top of CAT+NUM+DAY+ARR (the new
baseline, since arrmode passed docs/ARRMODE.md)? Same harness as arr_feat_test.py: CPU LightGBM,
physical class capped at PHYS_CAP rows/fold, one shared physical regressor per fold.
`PYTHONPATH=/c/Users/skf_s/prc26 python analysis/fe_test.py`"""
from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from data import CACHE, rmse
from day_feat_test import CLF, cap_phys, fit_mu
from fe import FP, G2, G3, build_fp, fit_apply, flight_ids, oof_apply
from features import ARR, CAT, DAY, NUM, build
from mix import Z, label
from train import FOLDS, HOLDOUT

VARIANTS = {"base": [], "fp": FP, "g2": G2, "g3": G3, "all": FP + G2 + G3}


def score(cols: list[str], tr: pd.DataFrame, lab_tr: np.ndarray, va: pd.DataFrame, lab_va: np.ndarray,
          mu_va: np.ndarray) -> dict[str, float]:
    """Fit the multiclass stamp classifier on `cols`, report log loss, mixture RMSE, class recall."""
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
    return dict(logloss=ll, rmse=rmse(va["y"].values, yhat), recall_eobt=recall_eobt, recall_sched=recall_sched)


def leak_check(tr_all: pd.DataFrame) -> tuple[float, float]:
    """fl_eobt_rate vs the row's own true-eobt label: in-fold (fit_apply on itself) should not blow past out-of-fold."""
    lab = label(tr_all)
    eobt_lab = (lab == 1).astype(int)
    infold = fit_apply(tr_all, tr_all)["fl_eobt_rate"].values
    oof = oof_apply(tr_all)["fl_eobt_rate"].values
    return float(np.corrcoef(infold, eobt_lab)[0, 1]), float(np.corrcoef(oof, eobt_lab)[0, 1])


def main() -> None:
    f = build().merge(build_fp(), on="id", how="left").merge(flight_ids(), on="id", how="left")
    tr_all = f[(f["split"] == "train") & ~f["month"].isin(HOLDOUT) & (f["has_plan"] == 1)]
    ho = f[(f["split"] == "train") & f["month"].isin(HOLDOUT) & (f["has_plan"] == 1)]
    base_cols = CAT + NUM + DAY + ARR

    corr_in, corr_out = leak_check(tr_all)
    print(f"leak check fl_eobt_rate vs true-eobt label: in-fold corr={corr_in:.4f}, out-of-fold corr={corr_out:.4f}")

    rows, t0 = [], time.time()
    for i, months in enumerate(FOLDS):
        va_m = tr_all["month"].isin(months).values
        tr, va = tr_all[~va_m], tr_all[va_m]
        lab_tr, lab_va = label(tr), label(va)
        mu_va = fit_mu(tr, lab_tr, va)
        tr_e = pd.concat([tr, oof_apply(tr)], axis=1)
        va_e = pd.concat([va, fit_apply(tr, va)], axis=1)
        for vname, extra in VARIANTS.items():
            m = score(base_cols + extra, tr_e, lab_tr, va_e, lab_va, mu_va)
            rows.append({"fold": f"fold{i}", "variant": vname, "n": len(va), **m})
        print(f"fold {i} {months}: " + " | ".join(f"{v}: rmse={r['rmse']:.2f} ll={r['logloss']:.4f}"
              for v, r in zip(VARIANTS, rows[-len(VARIANTS):])) + f"  ({time.time()-t0:.0f}s)", flush=True)

    lab_tr, lab_ho = label(tr_all), label(ho)
    mu_ho = fit_mu(tr_all, lab_tr, ho)
    tr_e = pd.concat([tr_all, oof_apply(tr_all)], axis=1)
    ho_e = pd.concat([ho, fit_apply(tr_all, ho)], axis=1)
    for vname, extra in VARIANTS.items():
        m = score(base_cols + extra, tr_e, lab_tr, ho_e, lab_ho, mu_ho)
        rows.append({"fold": "holdout", "variant": vname, "n": len(ho), **m})
    print("holdout: " + " | ".join(f"{v}: rmse={r['rmse']:.2f} ll={r['logloss']:.4f}"
          for v, r in zip(VARIANTS, rows[-len(VARIANTS):])) + f"  ({time.time()-t0:.0f}s)")

    out = pd.DataFrame(rows)
    out.to_csv(CACHE.parent / "analysis_fe_test.csv", index=False)
    print("\n", out.round(4).to_string(index=False))

    ho_rows = out[out["fold"] == "holdout"].set_index("variant")
    base_rmse = ho_rows.loc["base", "rmse"]
    fold_rows = out[out["fold"] != "holdout"]
    for v in ("fp", "g2", "g3", "all"):
        beats_every_fold = bool(np.all(fold_rows[fold_rows["variant"] == v]["rmse"].values <
                                        fold_rows[fold_rows["variant"] == "base"]["rmse"].values))
        delta = base_rmse - ho_rows.loc[v, "rmse"]
        passed = delta > 0 and beats_every_fold
        print(f"Test verdict [{v}]: holdout delta={delta:+.2f}, beats base every fold={beats_every_fold} -> "
              f"{'PASS' if passed else 'FAIL'}")


if __name__ == "__main__":
    main()
