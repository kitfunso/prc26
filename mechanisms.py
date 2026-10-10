"""Bounded, training-scope-only alternatives to Lane B's recording mixture."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import laneb
from data import CACHE
from mix import B_CAT, B_NUM


def copy_labels(rows: pd.DataFrame) -> np.ndarray:
    copy, near, _, _ = laneb.classes(rows)
    return (copy | near).astype(float)


def keyed(rows: pd.DataFrame) -> pd.DataFrame:
    raw = pd.read_parquet(CACHE / "raw.parquet", columns=["MVT_ID_mvt", "PHASE_mvt", "FLIGHT_mvt"])
    flights = raw[raw.PHASE_mvt == "DEP"].set_index("MVT_ID_mvt")["FLIGHT_mvt"]
    _, cell = laneb._cells(rows)
    cell = rows.apt.astype(str).values + "|" + cell
    return pd.DataFrame({"cell": cell, "airline": cell + "|" + rows.airline.astype(str).values,
                         "flight": cell + "|" + rows.id.map(flights).fillna("missing").astype(str).values}, index=rows.index)


def hierarchy(tr: pd.DataFrame, te: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    train, query = keyed(tr), keyed(te)
    train["copy"] = copy_labels(tr)
    rate, support = np.full(len(te), (train["copy"].sum() + 1) / (len(train) + 2)), np.zeros(len(te))
    for key, shrink in (("cell", 2.), ("airline", 20.), ("flight", 10.)):
        g = train.groupby(key)["copy"].agg(["sum", "size"])
        n = query[key].map(g["size"]).fillna(0).values
        rate = (query[key].map(g["sum"]).fillna(0).values + shrink * rate) / (n + shrink)
        support = n
    return rate, support


def classifier(tr: pd.DataFrame, te: pd.DataFrame) -> np.ndarray:
    cols = B_CAT + B_NUM + ["cross", "sched_hour"] + laneb.DAY_COLS
    model = lgb.LGBMClassifier(objective="binary", learning_rate=.05, num_leaves=15,
                              min_child_samples=60, n_estimators=120, reg_lambda=10,
                              random_state=0, n_jobs=8, verbose=-1)
    model.fit(tr[cols], copy_labels(tr), categorical_feature=B_CAT)
    return model.predict_proba(te[cols])[:, 1]


def calibrated(tr: pd.DataFrame, te: pd.DataFrame) -> tuple[np.ndarray, dict]:
    months = sorted(tr.month.unique())
    p = np.full(len(tr), np.nan)
    for i in range(3):
        va = tr.month.isin(months[i::3]).values
        p[va] = classifier(tr[~va], tr[va])
    logit = lambda x: np.log(np.clip(x, 1e-5, 1 - 1e-5) / np.clip(1 - x, 1e-5, 1))
    weight = np.clip(np.abs(tr.z_sched.values - 1100), 300, 60000) ** 2
    weight /= weight.mean()
    cal = LogisticRegression(C=10, random_state=0)
    cal.fit(logit(p)[:, None], copy_labels(tr), sample_weight=weight)
    raw = classifier(tr, te)
    q = cal.predict_proba(logit(raw)[:, None])[:, 1]
    return q, {"calibration_fit_rows": len(tr), "crossfit_month_groups": [months[i::3] for i in range(3)],
               "slope": float(cal.coef_[0, 0]), "intercept": float(cal.intercept_[0]),
               "raw_probabilities": raw.tolist(), "calibrated_probabilities": q.tolist()}


def outcomes(tr: pd.DataFrame, te: pd.DataFrame, control: np.ndarray) -> tuple[np.ndarray, dict]:
    lab = np.where(copy_labels(tr) > 0, 0, np.where(tr.y > 43200, 2, np.where(tr.y < -43200, 3, 1)))
    cols = B_CAT + B_NUM + ["cross", "sched_hour"] + laneb.DAY_COLS
    model = lgb.LGBMClassifier(learning_rate=.05, num_leaves=15, min_child_samples=60,
                              n_estimators=120, reg_lambda=10, random_state=0, n_jobs=8, verbose=-1)
    model.fit(tr[cols], lab, categorical_feature=B_CAT)
    probabilities = np.zeros((len(te), 4))
    probabilities[:, model.classes_] = model.predict_proba(te[cols])
    support = tr.assign(shift=(lab >= 2)).groupby("apt", observed=True)["shift"].sum()
    unsupported = te.apt.map(support).astype(float).fillna(0).values < 20
    probabilities[unsupported, 2:] = 0
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    phys = lab == 1
    tcols = laneb.TAU_CAT + laneb.tau_num(tr) + laneb.DAY + laneb.OCC + laneb.WX + laneb.ARR + laneb.WZ
    tau, _ = laneb.fit_tau(tr, phys, te, None, None, tcols, 300)
    p = probabilities[:, 0] * te.z_sched.values + probabilities[:, 1] * tau
    p += probabilities[:, 2] * (86400 + tau) + probabilities[:, 3] * (tau - 86400)
    # Sparse high-gap regions keep the established cell prior.
    _, cell = laneb._cells(te)
    counts = pd.Series(laneb._cells(tr)[1]).value_counts()
    sparse = (te.apt.values == "LIRF") & (te.z_sched.values >= 14400) & (pd.Series(cell).map(counts).values < 20)
    p[sparse] = control[sparse]
    return np.maximum(p, 0), {"classes": {str(k): int((lab == k).sum()) for k in np.unique(lab)},
                             "shift_airport_support": support.to_dict(), "sparse_control_rows": int(sparse.sum())}


def candidates(tr: pd.DataFrame, te: pd.DataFrame, control: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
    ht = (tr.apt == "LIRF") & (tr.z_sched >= 14400)
    he = (te.apt == "LIRF") & (te.z_sched >= 14400)
    q, support = hierarchy(tr[ht], te[he])
    kn, _ = laneb._cells(te[he])
    physical = 1100 + 86400 * kn
    hierarchical = control.copy()
    hierarchical[he] = q * te.loc[he, "z_sched"].values + (1 - q) * physical
    cal, detail = calibrated(tr, te)
    calibrated_p = control.copy()
    cell_train = pd.DataFrame({"cell": laneb._cells(tr[ht])[1], "copy": copy_labels(tr[ht])})
    rates = cell_train.groupby("cell")["copy"].agg(["sum", "size"])
    prior = pd.Series(laneb._cells(te[he])[1]).map((rates["sum"] + 1) / (rates["size"] + 2)).fillna(.5).values
    mixed = .5 * np.clip(prior, 0, 1) + .5 * cal[he]
    calibrated_p[he] = mixed * te.loc[he, "z_sched"].values + (1 - mixed) * physical
    outcome, outcome_detail = outcomes(tr, te, control)
    detail.update({"hierarchy_fit_rows": int(ht.sum()), "changed_region_rows": int(he.sum()),
                   "flight_support_min": float(support.min()) if len(support) else 0,
                   "flight_support_median": float(np.median(support)) if len(support) else 0,
                   "outcomes": outcome_detail})
    return {"CP1-hierarchy": laneb.hull(te, hierarchical), "CP1-calibration": laneb.hull(te, calibrated_p),
            "CP1-outcomes": outcome}, detail
