"""Lane A schedule-copy head: P(recorded off-block is a copy of the schedule) from copy history (laneb's `_hist_rates`
scored on lane A rows), the plan stamps and identity columns. recal.py's second stage takes P as a column (docs/RECAL.md).
2025 rows get P from models that never saw their month group or the holdout months; rank rows from one model on all of
2025. `python copyhead.py`."""
from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from data import CACHE
from features import build
from laneb import COPY_TOL, HIST, HIST_GAP, HIST_Z, _hist_rates
from mix import label
from train import FOLDS, HOLDOUT

C_CAT = ["apt", "airline", "stand", "runway", "actype", "segment", "ftype", "operator"]
C_NUM = HIST + ["z_sched", "z_aobt", "z_eobt", "d_as", "d_es", "d_ae", "hour", "wday", "occ", "t_occ", "queue", "dep_before", "notyp"]
CLF = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_child_samples=100, feature_fraction=0.8, bagging_fraction=0.8,
           bagging_freq=1, lambda_l2=3.0, n_estimators=3000, verbose=-1, random_state=0, n_jobs=16)


def hist_a(a: pd.DataFrame) -> pd.DataFrame:
    """laneb.copy_hist for lane A rows: 2025 rows see other non-holdout months HIST_GAP+ away (holdout rows see all ten),
    rank rows see all of 2025."""
    r = pd.read_parquet(CACHE / "raw.parquet", columns=["MVT_ID_mvt", "FLIGHT_mvt", "PHASE_mvt"])
    flt = r[r["PHASE_mvt"] == "DEP"].drop_duplicates("MVT_ID_mvt").set_index("MVT_ID_mvt")["FLIGHT_mvt"]
    apt = a["apt"].astype(str)
    a = a.assign(apt_s=apt, k_al=apt + "|" + a["airline"].astype(str), k_flt=apt + "|" + a["id"].map(flt).astype(str),
                 copy=(np.abs(a["y"] - a["z_sched"]) < COPY_TOL).astype(float))
    src = a[(a["split"] == "train") & (a["z_sched"] >= HIST_Z)]
    src_tr, out = src[~src["month"].isin(HOLDOUT)], pd.DataFrame(np.nan, index=a.index, columns=HIST)
    for (split, m), q in a.groupby(["split", "month"], observed=True):
        h = src if split == "rank" else src_tr if m in HOLDOUT else src_tr[(src_tr["month"] - m).abs() >= HIST_GAP]
        out.loc[q.index] = _hist_rates(h, q)
    return out


def fit_p(trn: pd.DataFrame, te: pd.DataFrame) -> tuple[np.ndarray, int]:
    es = pd.factorize(trn["day"])[0] % 10 == 0
    m = lgb.LGBMClassifier(**CLF)
    m.fit(trn[C_CAT + C_NUM][~es], trn["sc"][~es], eval_set=[(trn[C_CAT + C_NUM][es], trn["sc"][es])], categorical_feature=C_CAT,
          callbacks=[lgb.early_stopping(100, verbose=False)])
    return m.predict_proba(te[C_CAT + C_NUM])[:, 1], m.best_iteration_


def main() -> None:
    t0, f = time.time(), build()
    a = f[(f["has_plan"] == 1) & f["split"].isin(("train", "rank"))].copy()
    del f
    a[HIST] = hist_a(a)
    tr = (a["split"] == "train").values
    a["sc"] = 0
    a.loc[tr, "sc"] = (label(a[tr]) == 2).astype(int)
    ho, P = tr & a["month"].isin(HOLDOUT).values, np.full(len(a), np.nan)
    for g in FOLDS + (HOLDOUT, None):
        te = ~tr if g is None else tr & a["month"].isin(g).values
        trn = tr if g is None else tr & ~ho & ~te
        P[te], it = fit_p(a[trn], a[te])
        auc = "" if g is None else f", AUC {roc_auc_score(a['sc'].values[te], P[te]):.3f}"
        print(f"P(sched copy) for {'rank' if g is None else g}: {it} iters{auc}, {time.time() - t0:.0f}s", flush=True)
    assert not np.isnan(P).any()
    pd.DataFrame({"id": a["id"].values, "P": P}).to_parquet(CACHE / "copyhead.parquet")


if __name__ == "__main__":
    main()
