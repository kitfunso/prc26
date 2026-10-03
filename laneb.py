"""Lane B mechanism model: y is one of {schedule copy, day-shifted physical, physical} (near folded into physical,
see docs/LANEB.md), not a single donor regression. The classifier also sees how often the row's airline and flight
number copy the schedule in lane A (`copy_hist()`). Ambiguous copy rows blend toward the donor's cached prediction;
LIRF rows over 4 h late take a per-cell copy-share prior instead. `python laneb.py cv DONOR` / `python laneb.py fit IT_CLF IT_TAU [DONOR]`."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from data import CACHE, rmse
from features import ARR, DAY, NUM, OCC, WX, WZ, build
from mix import B_CAT, B_NUM
from train import FOLDS, HOLDOUT, LGB, RESULTS, boot_delta, report, save_model

DOC = "docs/LANEB_RUN.md"
BLEND = CACHE / "blend_ra_cb+ra_cb_s1+ra_cb_s2+ra_lgb+nn3_holdout.parquet"
APT_SHIFT, Z_SHIFT, Y_SHIFT, SHIFT_ADD = "LIRF", 57600.0, 72000.0, 86400.0
COPY_TOL, NEAR_Y_MIN, NEAR_LO, NEAR_HI, WIN = 60.0, 3000.0, -700.0, 3100.0, 7200.0
DONOR_COPY_THRESH, DONOR_COPY_W = 0.12, 0.10
W_FLOOR, W_CAP = 300.0, 14400.0
HULL_Z, HULL_TAU, HULL_TZ, PRIOR_TAU = 14400.0, 3600.0, "Europe/Rome", 1100.0
TAU_CAT = ["apt", "runway", "stand", "actype", "airline", "ades"]
DAY_COLS = ["est_share", "a_tail_share", "a_za_p90", "w_est_share", "w_tail_share", "w_za_p90"]
HIST, HIST_Z, HIST_GAP, HIST_K = ["h_al", "h_flt", "h_nflt"], 1800.0, 3, (20.0, 5.0)
CLF_LGB = dict(objective="multiclass", learning_rate=0.05, num_leaves=63, min_child_samples=30, feature_fraction=0.8,
               bagging_fraction=0.8, bagging_freq=1, lambda_l2=3.0, n_estimators=2000, verbose=-1, random_state=0, n_jobs=8)


def classes(d: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """copy: within 60s of schedule. shift: 16 rows where the recorder stamped tomorrow's clock time (own rule cell).
    near: stamp 0-25 min after schedule, takeoff hours later (recorder artefact, not congestion). phys: the rest."""
    y, zs = d["y"].values, d["z_sched"].values
    copy = np.abs(y - zs) < COPY_TOL
    shift = y > Y_SHIFT
    near = ~copy & ~shift & (y > NEAR_Y_MIN) & ((zs - y) >= NEAR_LO) & ((zs - y) <= NEAR_HI)
    return copy, near, ~copy & ~shift & ~near, shift


def _labels(d: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """copy vs phys label (near folded into phys, it never earned a separate class); shift excluded from training."""
    copy, _, _, shift = classes(d)
    return np.where(copy, 0, 2), ~shift, ~copy & ~shift


def _window(f: pd.DataFrame, a: pd.DataFrame, near: np.ndarray, tail: np.ndarray) -> pd.DataFrame:
    """+-2h share/p90 around each lane-B row's own takeoff, from lane-A rows of the same airport only."""
    out = {c: np.full(len(f), np.nan) for c in ("w_est_share", "w_tail_share", "w_za_p90")}
    b = f[f["has_plan"] == 0]
    keys = ["file", "apt"] if "file" in f else "apt"
    for key, ix in b.groupby(keys, observed=True).indices.items():
        apt = key[1] if "file" in f else key
        m = (a["apt"].values == apt)
        if "file" in f:
            m &= a["file"].values == key[0]
        order = np.argsort(a["t_off"].values[m])
        ta, za = a["t_off"].values[m][order], a["z_aobt"].values[m][order]
        na, ta_ = near[m][order], tail[m][order]
        q = b["t_off"].values[ix]
        lo, hi = np.searchsorted(ta, q - WIN, "left"), np.searchsorted(ta, q + WIN, "right")
        for k, p in enumerate(b.index.values[ix]):
            l, h = lo[k], hi[k]
            if h > l:
                out["w_est_share"][p], out["w_tail_share"][p] = na[l:h].mean(), ta_[l:h].mean()
                out["w_za_p90"][p] = np.percentile(za[l:h], 90)
    return pd.DataFrame(out, index=f.index)


def augment(f: pd.DataFrame) -> pd.DataFrame:
    """Cross-midnight flag, schedule hour, and lane-A airport-day signals visible in both splits (no label leak)."""
    sched = f["t_off"].values - f["z_sched"].values
    f = f.assign(cross=((sched // 86400) < (f["t_off"].values // 86400)).astype(int),
                 sched_hour=((sched % 86400) // 3600).astype(float))
    a = f[f["has_plan"] == 1]
    za, zs = a["z_aobt"].values, a["z_sched"].values
    near_a, tail_a = (za > 3600) & ((zs - za) >= NEAR_LO) & ((zs - za) <= NEAR_HI), za > 3600
    keys = (["file"] if "file" in f else []) + ["apt", "day"]
    day = a.assign(near=near_a, tail=tail_a).groupby(keys, observed=True).agg(
        est_share=("near", "mean"), a_tail_share=("tail", "mean"), a_za_p90=("z_aobt", lambda s: s.quantile(.9))).reset_index()
    f = f.merge(day, on=keys, how="left")
    return pd.concat([f, _window(f, a, near_a, tail_a)], axis=1)


def _hist_rates(a: pd.DataFrame, q: pd.DataFrame) -> np.ndarray:
    """Copy share of a's rows by (airport, airline), shrunk toward the airport share, and by (airport, flight number),
    shrunk toward the airline share; plus log(1 + rows behind the flight share)."""
    if a.empty:
        return np.tile((0.5, 0.5, 0.0), (len(q), 1))
    p0 = q["apt_s"].map(a.groupby("apt_s")["copy"].mean()).fillna(a["copy"].mean()).values
    s_al, s_flt = (a.groupby(k)["copy"].agg(["sum", "size"]) for k in ("k_al", "k_flt"))
    r_al = (q["k_al"].map(s_al["sum"]).fillna(0).values + HIST_K[0] * p0) / (q["k_al"].map(s_al["size"]).fillna(0).values + HIST_K[0])
    n_flt = q["k_flt"].map(s_flt["size"]).fillna(0).values
    r_flt = (q["k_flt"].map(s_flt["sum"]).fillna(0).values + HIST_K[1] * r_al) / (n_flt + HIST_K[1])
    return np.column_stack([r_al, r_flt, np.log1p(n_flt)])


def copy_hist(f: pd.DataFrame, fit_ids: pd.Index) -> pd.DataFrame:
    """Only permitted departure labels feed history; the month gap also cross-fits training rows."""
    fit_ids = pd.Index(fit_ids)
    if not fit_ids.is_unique or not fit_ids.isin(f.loc[f["split"] == "train", "id"]).all():
        raise ValueError("history fit IDs must be unique training departure IDs")
    r = pd.read_parquet(CACHE / "raw.parquet", columns=["MVT_ID_mvt", "FLIGHT_mvt", "PHASE_mvt"])
    flt = r[r["PHASE_mvt"] == "DEP"].drop_duplicates("MVT_ID_mvt").set_index("MVT_ID_mvt")["FLIGHT_mvt"]
    source = f["id"].isin(fit_ids) & (f["has_plan"] == 1) & (f["z_sched"] >= HIST_Z)
    keys = f.loc[source | (f["has_plan"] == 0), ["id", "apt", "airline", "split", "month"]].copy()
    apt = keys["apt"].astype(str)
    keys = keys.assign(apt_s=apt, k_al=apt + "|" + keys["airline"].astype(str),
                       k_flt=apt + "|" + keys["id"].map(flt).astype(str))
    a = keys.loc[f.index[source]].assign(copy=(np.abs(f.loc[source, "y"] - f.loc[source, "z_sched"]) < COPY_TOL).astype(float))
    b, out = keys.loc[f.index[f["has_plan"] == 0]], pd.DataFrame(np.nan, index=f.index, columns=HIST)
    for (split, m), q in b.groupby(["split", "month"], observed=True):
        out.loc[q.index] = _hist_rates(a if split == "rank" else a[(a["month"] - m).abs() >= HIST_GAP], q)
    return out


def history_rows(f: pd.DataFrame, fit_ids: pd.Index) -> pd.DataFrame:
    b = f[f["has_plan"] == 0]
    return pd.concat([b, copy_hist(f, fit_ids).loc[b.index]], axis=1)


def stopping_split(tr: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    days = np.sort(tr["day"].unique())
    if len(days) < 2:
        raise ValueError("iteration selection needs at least two training days")
    stop = tr["day"].isin(days[-max(1, int(np.ceil(len(days) * 0.1))):])
    return tr[~stop], tr[stop]


def outer_predict(f: pd.DataFrame, fit_mask: pd.Series, eval_ids: pd.Index) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, tuple[int, int]]:
    fit_ids = pd.Index(f.loc[fit_mask, "id"])
    if fit_ids.isin(eval_ids).any():
        raise ValueError("outer evaluation IDs overlap training IDs")
    scoped = history_rows(f, fit_ids)
    trn = scoped[scoped["id"].isin(fit_ids)]
    van = scoped[scoped["id"].isin(eval_ids)]
    inner_train, inner_stop = stopping_split(trn)
    inner = fit_mask & ~f["day"].isin(inner_stop["day"].unique())
    selected = history_rows(f, pd.Index(f.loc[inner, "id"]))
    _, it, *_ = predict(selected.loc[inner_train.index], selected.loc[inner_stop.index], selected.loc[inner_stop.index])
    p, _, *_ = predict(trn, van, iters=it)
    return trn, van, p, it


def shift_rate(tr: pd.DataFrame) -> tuple[float, int]:
    sub = tr[(tr["apt"] == APT_SHIFT) & (tr["z_sched"] > Z_SHIFT)]
    return (float((sub["y"] > Y_SHIFT).mean()) if len(sub) else 0.0), len(sub)


def tau_num(b: pd.DataFrame) -> list[str]:
    """NUM columns that are structurally NaN on lane B (AOBT/EOBT/LOBT/IOBT derived) carry no signal here."""
    return [c for c in NUM if b[c].notna().any()]


def _cost_w(d: pd.DataFrame, tau0: pd.Series) -> np.ndarray:
    """A wrong copy/phys call costs about (z_sched - tau)^2 of SSE, so rows weigh |z_sched - tau0|, clipped, mean 1."""
    t0 = d["apt"].astype(str).map(tau0).astype(float).fillna(float(tau0.median())).values
    w = np.clip(np.abs(d["z_sched"].values - t0), W_FLOOR, W_CAP)
    return w / w.mean()


def fit_clf(tr, lab, keep, te, va, vlab, vkeep, cols, iters, model_path: Path | None = None) -> tuple[np.ndarray, int]:
    """Multiclass over whichever of {copy, near, phys} are present in tr, expanded back to a fixed 3-column P. Rows are
    cost-weighted (`_cost_w`, tau0 = the airport's median physical y in tr), see docs/LANEB.md section 5."""
    present = sorted(set(lab[keep].tolist()))
    remap = {v: i for i, v in enumerate(present)}
    y = np.vectorize(remap.get)(lab[keep])
    tau0 = tr[keep & (lab == 2)].groupby(tr["apt"].astype(str))["y"].median()
    params = {**CLF_LGB, "num_class": len(present)}
    n = iters or params.pop("n_estimators")
    m = lgb.LGBMClassifier(**{**params, "n_estimators": n})
    if va is not None:
        yv = np.vectorize(remap.get)(vlab[vkeep])
        m.fit(tr[cols][keep], y, sample_weight=_cost_w(tr[keep], tau0), eval_set=[(va[cols][vkeep], yv)],
              eval_sample_weight=[_cost_w(va[vkeep], tau0)], categorical_feature=B_CAT, callbacks=[lgb.early_stopping(100, verbose=False)])
        it = m.best_iteration_ or n
    else:
        m.fit(tr[cols][keep], y, sample_weight=_cost_w(tr[keep], tau0), categorical_feature=B_CAT); it = n
    save_model(m, model_path, "lgb")
    raw, P = m.predict_proba(te[cols]), np.zeros((len(te), 3))
    for v, i in remap.items():
        P[:, v] = raw[:, i]
    return P, it


def fit_tau(tr, phys, te, va, vphys, cols, iters, model_path: Path | None = None) -> tuple[np.ndarray, int]:
    params = dict(LGB); n = iters or params.pop("n_estimators")
    m = lgb.LGBMRegressor(**{**params, "n_estimators": n})
    if va is not None:
        m.fit(tr[cols][phys], tr["y"][phys], eval_set=[(va[cols][vphys], va["y"][vphys])], categorical_feature=TAU_CAT,
              callbacks=[lgb.early_stopping(200, verbose=False)])
        it = m.best_iteration_ or n
    else:
        m.fit(tr[cols][phys], tr["y"][phys], categorical_feature=TAU_CAT); it = n
    save_model(m, model_path, "lgb")
    return m.predict(te[cols]), it


def clf_logloss(tr: pd.DataFrame, va: pd.DataFrame, use_day: bool, iters: int) -> tuple[float, int]:
    """Classifier-only fit (no tau) to isolate the day features' effect on multiclass log loss."""
    lab, keep, _ = _labels(tr)
    vlab, vkeep, _ = _labels(va)
    cols = B_CAT + B_NUM + ["cross", "sched_hour"] + HIST + (DAY_COLS if use_day else [])
    P, _ = fit_clf(tr, lab, keep, va, None, None, None, cols, iters)
    p = np.clip(P[vkeep], 1e-9, 1); p /= p.sum(1, keepdims=True)
    return log_loss(vlab[vkeep], p, labels=[0, 1, 2]), int(vkeep.sum())


def predict(tr, te, va=None, iters=None, use_shift=True, use_day=True, use_hist=True, model_dir: Path | None = None):
    """tr/te/va: lane-B rows only. yhat = P(copy)*z_sched + P(phys)*tau, shift rows get 86400+tau on top."""
    lab, keep, phys = _labels(tr)
    clf_cols = B_CAT + B_NUM + ["cross", "sched_hour"] + (HIST if use_hist else []) + (DAY_COLS if use_day else [])
    vlab = vkeep = vphys = None
    if va is not None:
        vlab, vkeep, vphys = _labels(va)
    gate_path = model_dir / "gate.model" if model_dir else None
    tau_path = model_dir / "tau.model" if model_dir else None
    P, it_c = fit_clf(tr, lab, keep, te, va, vlab, vkeep, clf_cols, iters[0] if iters else None, model_path=gate_path)
    tcols = TAU_CAT + tau_num(tr) + DAY + OCC + WX + ARR + WZ
    tau, it_t = fit_tau(tr, phys, te, va, vphys, tcols, iters[1] if iters else None, model_path=tau_path)
    rate, n_shift = shift_rate(tr)
    p_shift = np.where(use_shift & (te["apt"].values == APT_SHIFT) & (te["z_sched"].values > Z_SHIFT), rate, 0.0)
    Psum = P.sum(1, keepdims=True); Psum[Psum == 0] = 1.0; P = P / Psum
    rest = 1 - p_shift
    yhat = rest * (P[:, 0] * te["z_sched"].values + P[:, 2] * tau) + p_shift * (SHIFT_ADD + tau)
    return np.maximum(yhat, 0), (it_c, it_t), P, rate, n_shift


def blend_donor(p_new: np.ndarray, p_copy: np.ndarray, donor_p: np.ndarray) -> np.ndarray:
    """Ambiguous copy rows (P(copy) > DONOR_COPY_THRESH): our tau misjudges the physical remainder there, so blend
    toward the donor's cached prediction (see docs/LANEB.md) instead of trusting ours alone."""
    donor_p = np.where(np.isnan(donor_p), p_new, donor_p); mixed = DONOR_COPY_W * p_new + (1 - DONOR_COPY_W) * donor_p
    return np.maximum(np.where(p_copy > DONOR_COPY_THRESH, mixed, p_new), 0)


def _lday(t: np.ndarray) -> np.ndarray:
    return pd.to_datetime(t, unit="s", utc=True).tz_convert(HULL_TZ).tz_localize(None).values.astype("datetime64[D]").astype(np.int64)


def hull(d: pd.DataFrame, p: np.ndarray) -> np.ndarray:
    """LIRF rows over 4h late: y is a schedule copy (z or the near band) or the off-block clock time pinned to date k, the
    date nearest the schedule or the local date, so p goes onto the convex hull of those outcomes (docs/LANEB.md)."""
    z, t = d["z_sched"].values, d["t_off"].values
    k = np.array([np.round((z - a) / SHIFT_ADD) for a in (0.0, HULL_TAU)] + [_lday(t - a) - _lday(t - z) for a in (0.0, HULL_TAU)])
    lo, hi = np.minimum(z - NEAR_HI, SHIFT_ADD * k.min(0)), np.maximum(z - NEAR_LO, HULL_TAU + SHIFT_ADD * k.max(0))
    return np.where((d["apt"].values == APT_SHIFT) & (z >= HULL_Z), np.clip(p, lo, hi), p)


def _cells(d: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """kn: whole days of delay after a PRIOR_TAU taxi; kd: local dates between the schedule and that off-block. kn != kd
    only fits a copy (cell D); the rest split by kn = 0 (A0) and z_sched either side of Z_SHIFT (A1lo, A1hi)."""
    z, t = d["z_sched"].values, d["t_off"].values
    kn, kd = np.round((z - PRIOR_TAU) / SHIFT_ADD), _lday(t - PRIOR_TAU) - _lday(t - z)
    return kn, np.where(kn != kd, "D", np.where(kn == 0, "A0", np.where(z > Z_SHIFT, "A1hi", "A1lo")))


def hull_prior(tr: pd.DataFrame, te: pd.DataFrame, p: np.ndarray) -> np.ndarray:
    """LIRF rows over 4h late: the classifier's P(copy) does not sort copies there, so p becomes P(copy) z + (1 - P(copy))
    (PRIOR_TAU + 86400 kn), P(copy) the Laplace-smoothed copy share of tr's rows in the same cell (docs/LANEB.md section 5)."""
    ht = (tr["apt"].values == APT_SHIFT) & (tr["z_sched"].values >= HULL_Z)
    he = (te["apt"].values == APT_SHIFT) & (te["z_sched"].values >= HULL_Z)
    c, nr, _, _ = classes(tr[ht])
    _, cell_tr = _cells(tr[ht])
    kn, cell = _cells(te[he])
    pc = np.array([((c | nr)[cell_tr == k].sum() + 1) / ((cell_tr == k).sum() + 2) for k in cell])
    out = np.array(p, dtype=float)
    out[he] = pc * te["z_sched"].values[he] + (1 - pc) * (PRIOR_TAU + SHIFT_ADD * kn)
    return out


def overall(f: pd.DataFrame, ho: pd.DataFrame, ph: np.ndarray) -> tuple[float, float]:
    """Holdout RMSE over all 344,419 rows: donor blend as-is, vs. the same blend with lane B rows replaced."""
    bl = pd.read_parquet(BLEND)
    y = f[(f["split"] == "train") & f["month"].isin(HOLDOUT)][["id", "y"]]
    base = bl.merge(y, on="id")
    new = bl.set_index("id")["p"].copy()
    new.loc[ho["id"].values] = ph
    swapped = y.merge(new.rename("p").reset_index(), on="id")
    return rmse(base["y"].values, base["p"].values), rmse(swapped["y"].values, swapped["p"].values)


def cv(donor: str) -> None:
    f = augment(build())
    development = (f["split"] == "train") & ~f["month"].isin(HOLDOUT)
    b = history_rows(f, pd.Index(f.loc[development, "id"]))
    tr_all, ho = b[(b["split"] == "train") & ~b["month"].isin(HOLDOUT)], b[(b["split"] == "train") & b["month"].isin(HOLDOUT)]
    copy, near, phys, shift = classes(tr_all)
    print(f"flight copy history on lane B rows: train {np.mean(tr_all['h_nflt'] > 0):.3f}, holdout {np.mean(ho['h_nflt'] > 0):.3f}")
    print(pd.DataFrame({"apt": tr_all["apt"].values, "copy": copy, "near": near, "phys": phys, "shift": shift}).groupby("apt").sum())
    oof, its, t0 = np.full(len(tr_all), np.nan), [], time.time(); ll_day = ll_noday = n_ll = 0
    for i, months in enumerate(FOLDS):
        va = tr_all["month"].isin(months).values
        outer = development & ~f["month"].isin(months)
        trn, van, p, it = outer_predict(f, outer, pd.Index(tr_all.loc[va, "id"]))
        oof[va] = p = hull(van, hull_prior(trn, van, p)); its.append(it)
        l1, n1 = clf_logloss(trn, van, True, it[0]); l0, n0 = clf_logloss(trn, van, False, it[0])
        ll_day += l1 * n1; ll_noday += l0 * n0; n_ll += n1
        print(f"fold {i} {months}: rmse {rmse(van['y'].values, p):.1f} at {it} iters, {time.time() - t0:.0f}s", flush=True)
    print(f"day-feature multiclass logloss: with {ll_day / n_ll:.4f}, without {ll_noday / n_ll:.4f}")
    it = tuple(int(np.mean([x[j] for x in its]) * 1.1) for j in range(2))
    raw, _, P, rate, n_shift = predict(tr_all, ho, iters=it); print(f"shift rule: {APT_SHIFT} z_sched>{Z_SHIFT:.0f}, n={n_shift}, rate={rate:.3f}")
    dh = pd.read_parquet(CACHE / f"{donor}_holdout.parquet")[["id", "p"]].rename(columns={"p": "donor"})
    donor_p = ho[["id"]].merge(dh, on="id", how="left")["donor"].values; bl = blend_donor(raw, P[:, 0], donor_p)
    ph = hull(ho, hull_prior(tr_all, ho, bl))
    pd.DataFrame({"id": tr_all["id"].values, "oof": oof}).to_parquet(CACHE / "laneb_oof.parquet")
    pd.DataFrame({"id": ho["id"].values, "p": ph}).to_parquet(CACHE / "laneb_holdout.parquet")
    ho = ho.assign(p=ph); hb = ho.merge(dh, on="id")
    rmse_new, rmse_donor = rmse(hb["y"].values, hb["p"].values), rmse(hb["y"].values, hb["donor"].values)
    d = boot_delta(hb["y"].values, hb["donor"].values, hb["p"].values, hb["day"].values)
    abl = {"full": rmse_new, "donor_only": rmse_donor, "no_blend": rmse(ho["y"].values, hull(ho, hull_prior(tr_all, ho, raw))),
           "no_prior": rmse(ho["y"].values, hull(ho, bl))}
    for tag, kw in (("no_shift", dict(use_shift=False)), ("no_day", dict(use_day=False)), ("no_hist", dict(use_hist=False))):
        p, _, Pk, _, _ = predict(tr_all, ho, iters=it, **kw)
        abl[tag] = rmse(ho["y"].values, hull(ho, hull_prior(tr_all, ho, blend_donor(p, Pk[:, 0], donor_p))))
    base_rmse, new_rmse = overall(f, ho, ph)
    by_apt = pd.DataFrame({"apt": hb["apt"].values, "y": hb["y"].values, "new": hb["p"].values, "donor": hb["donor"].values}) \
        .groupby("apt").apply(lambda g: pd.Series({"n": len(g), "rmse_new": rmse(g["y"], g["new"]), "rmse_donor": rmse(g["y"], g["donor"])}), include_groups=False)
    mech = pd.Series("phys", index=hb.index)
    hc, hn, hp, hs = classes(hb); mech[hc], mech[hn], mech[hs] = "copy", "near", "shift"
    by_cls = pd.DataFrame({"mech": mech.values, "y": hb["y"].values, "new": hb["p"].values, "donor": hb["donor"].values}) \
        .groupby("mech").apply(lambda g: pd.Series({"n": len(g), "rmse_new": rmse(g["y"], g["new"]), "rmse_donor": rmse(g["y"], g["donor"])}), include_groups=False)
    print(f"holdout lane B rmse: new {rmse_new:.1f} vs donor {rmse_donor:.1f}, delta {d[0]:+.1f} [{d[1]:+.1f}, {d[2]:+.1f}]")
    print(by_apt.round(1).to_string()); print(by_cls.round(1).to_string())
    print(f"ablation: {abl}"); print(f"overall holdout (344,419 rows): donor blend {base_rmse:.2f} -> lane B swapped {new_rmse:.2f}")
    line = f"| laneb (donor {donor}) | - | {json.dumps(report(hb['y'].values, hb['p'].values, hb))} | {it} | {time.time() - t0:.0f}s |\n"
    RESULTS.open("a", encoding="utf-8").write(line)
    write_doc(donor, it, rate, n_shift, ll_day / n_ll, ll_noday / n_ll, rmse_new, rmse_donor, d, by_apt, by_cls, abl, base_rmse, new_rmse)


def write_doc(donor, it, rate, n_shift, ll_d, ll_nd, rmse_new, rmse_donor, d, by_apt, by_cls, abl, base_rmse, new_rmse) -> None:
    copy_new, copy_donor = by_cls.loc["copy", "rmse_new"], by_cls.loc["copy", "rmse_donor"]
    passed = rmse_new < 1900.6 and d[1] > 0 and copy_new <= copy_donor
    lines = [
        "# Lane B mechanism model: last `laneb.py cv` run (the analysis is docs/LANEB.md)\n\n",
        f"`PYTHONPATH=/c/Users/skf_s/prc26 python laneb.py cv {donor}` then `python laneb.py fit {it[0]} {it[1]}`.\n\n",
        f"Result: **{'PASS' if passed else 'FAIL'}** vs donor {donor} (1900.6). New {rmse_new:.1f}, donor {rmse_donor:.1f}, "
        f"bootstrap delta {d[0]:+.1f} [{d[1]:+.1f}, {d[2]:+.1f}]. Copy {copy_new:.1f} vs donor {copy_donor:.1f} ({'no worse' if copy_new <= copy_donor else 'worse'}).\n\n",
        f"Shift rule: {APT_SHIFT} z_sched > {Z_SHIFT:.0f}s, n={n_shift} train rows, empirical rate {rate:.3f}.\n\n",
        f"Day-feature multiclass log loss on CV folds: with {ll_d:.4f}, without {ll_nd:.4f} "
        f"({'day features help' if ll_d < ll_nd else 'day features do not help'}).\n\n",
        f"Donor blend: P(copy) > {DONOR_COPY_THRESH} rows take {DONOR_COPY_W:.2f}*ours + {1 - DONOR_COPY_W:.2f}*donor (tau misjudges the remainder there).\n\n",
        f"Hull: {APT_SHIFT} rows with z_sched >= {HULL_Z:.0f}s are projected onto the convex hull of their feasible labels (`hull()`).\n\n",
        f"Hull prior: before `hull()` those rows take P(copy) z + (1 - P(copy)) ({PRIOR_TAU:.0f} + 86400 kn), P(copy) the Laplace-smoothed "
        "copy share of their cell among the training rows (`hull_prior()`).\n\n",
        "## By airport (holdout lane B)\n\n", by_apt.round(1).to_markdown(), "\n\n",
        "## By mechanism class (holdout lane B)\n\n", by_cls.round(1).to_markdown(), "\n\n",
        "## Ablation (holdout lane B RMSE, one piece removed)\n\n",
        "| variant | rmse |\n|---|---|\n",
        *[f"| {k} | {v:.1f} |\n" for k, v in abl.items()],
        f"\nOverall holdout (344,419 rows, lane A unchanged): donor blend {base_rmse:.2f} -> lane B swapped {new_rmse:.2f}.\n",
    ]
    with open(DOC, "w", encoding="utf-8") as fh:
        fh.writelines(lines)


def fit(it_clf: int, it_tau: int, donor: str = "mix_cb_wx") -> None:
    f = augment(build())
    b = history_rows(f, pd.Index(f.loc[f["split"] == "train", "id"]))
    tr, rk = b[b["split"] == "train"], b[b["split"] == "rank"]
    p, _, P, _, _ = predict(tr, rk, iters=(it_clf, it_tau))
    dr = pd.read_parquet(f"data/{donor}_rank.parquet")[["MVT_ID_mvt", "TAXITIME_SEC_mvt"]].rename(columns={"MVT_ID_mvt": "id", "TAXITIME_SEC_mvt": "donor"})
    donor_p = rk[["id"]].merge(dr, on="id", how="left")["donor"].values; p = hull(rk, hull_prior(tr, rk, blend_donor(p, P[:, 0], donor_p)))
    pd.DataFrame({"MVT_ID_mvt": rk["id"].values, "TAXITIME_SEC_mvt": p}).to_parquet("data/laneb_rank.parquet")
    print("data/laneb_rank.parquet", len(p), float(np.mean(p)))


if __name__ == "__main__":
    {"cv": lambda: cv(sys.argv[2]), "fit": lambda: fit(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4] if len(sys.argv) > 4 else "mix_cb_wx")}[sys.argv[1]]()
