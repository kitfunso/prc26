"""Lane B mechanism model: y is one of {schedule copy, day-shifted physical, physical} (near folded into physical,
see docs/LANEB.md), not a single donor regression. Ambiguous copy rows blend toward the donor's cached prediction.
`python laneb.py cv DONOR` / `python laneb.py fit IT_CLF IT_TAU [DONOR]`."""
from __future__ import annotations

import json
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from data import CACHE, rmse
from features import DAY, NUM, OCC, WX, build
from mix import B_CAT, B_NUM
from train import FOLDS, HOLDOUT, LGB, RESULTS, boot_delta, report

DOC = "docs/LANEB.md"
BLEND = CACHE / "blend_ra_cb+ra_cb_s1+ra_cb_s2+ra_lgb+nn3_holdout.parquet"
APT_SHIFT, Z_SHIFT, Y_SHIFT, SHIFT_ADD = "LIRF", 57600.0, 72000.0, 86400.0
COPY_TOL, NEAR_Y_MIN, NEAR_LO, NEAR_HI, WIN = 60.0, 3000.0, -700.0, 3100.0, 7200.0
DONOR_COPY_THRESH, DONOR_COPY_W = 0.12, 0.10
TAU_CAT = ["apt", "runway", "stand", "actype", "airline", "ades"]
DAY_COLS = ["est_share", "a_tail_share", "a_za_p90", "w_est_share", "w_tail_share", "w_za_p90"]
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
    for apt, ix in b.groupby("apt", observed=True).indices.items():
        m = (a["apt"].values == apt)
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
    day = a.assign(near=near_a, tail=tail_a).groupby(["apt", "day"], observed=True).agg(
        est_share=("near", "mean"), a_tail_share=("tail", "mean"), a_za_p90=("z_aobt", lambda s: s.quantile(.9))).reset_index()
    f = f.merge(day, on=["apt", "day"], how="left")
    return pd.concat([f, _window(f, a, near_a, tail_a)], axis=1)


def shift_rate(tr: pd.DataFrame) -> tuple[float, int]:
    sub = tr[(tr["apt"] == APT_SHIFT) & (tr["z_sched"] > Z_SHIFT)]
    return (float((sub["y"] > Y_SHIFT).mean()) if len(sub) else 0.0), len(sub)


def tau_num(b: pd.DataFrame) -> list[str]:
    """NUM columns that are structurally NaN on lane B (AOBT/EOBT/LOBT/IOBT derived) carry no signal here."""
    return [c for c in NUM if b[c].notna().any()]


def fit_clf(tr, lab, keep, te, va, vlab, vkeep, cols, iters) -> tuple[np.ndarray, int]:
    """Multiclass over whichever of {copy, near, phys} are present in tr, expanded back to a fixed 3-column P."""
    present = sorted(set(lab[keep].tolist()))
    remap = {v: i for i, v in enumerate(present)}
    y = np.vectorize(remap.get)(lab[keep])
    params = {**CLF_LGB, "num_class": len(present)}
    n = iters or params.pop("n_estimators")
    m = lgb.LGBMClassifier(**{**params, "n_estimators": n})
    if va is not None:
        yv = np.vectorize(remap.get)(vlab[vkeep])
        m.fit(tr[cols][keep], y, eval_set=[(va[cols][vkeep], yv)], categorical_feature=B_CAT,
              callbacks=[lgb.early_stopping(100, verbose=False)])
        it = m.best_iteration_ or n
    else:
        m.fit(tr[cols][keep], y, categorical_feature=B_CAT); it = n
    raw, P = m.predict_proba(te[cols]), np.zeros((len(te), 3))
    for v, i in remap.items():
        P[:, v] = raw[:, i]
    return P, it


def fit_tau(tr, phys, te, va, vphys, cols, iters) -> tuple[np.ndarray, int]:
    params = dict(LGB); n = iters or params.pop("n_estimators")
    m = lgb.LGBMRegressor(**{**params, "n_estimators": n})
    if va is not None:
        m.fit(tr[cols][phys], tr["y"][phys], eval_set=[(va[cols][vphys], va["y"][vphys])], categorical_feature=TAU_CAT,
              callbacks=[lgb.early_stopping(200, verbose=False)])
        it = m.best_iteration_ or n
    else:
        m.fit(tr[cols][phys], tr["y"][phys], categorical_feature=TAU_CAT); it = n
    return m.predict(te[cols]), it


def clf_logloss(tr: pd.DataFrame, va: pd.DataFrame, use_day: bool) -> tuple[float, int]:
    """Classifier-only fit (no tau) to isolate the day features' effect on multiclass log loss."""
    lab, keep, _ = _labels(tr)
    vlab, vkeep, _ = _labels(va)
    cols = B_CAT + B_NUM + ["cross", "sched_hour"] + (DAY_COLS if use_day else [])
    P, _ = fit_clf(tr, lab, keep, va, va, vlab, vkeep, cols, None)
    p = np.clip(P[vkeep], 1e-9, 1); p /= p.sum(1, keepdims=True)
    return log_loss(vlab[vkeep], p, labels=[0, 1, 2]), int(vkeep.sum())


def predict(tr, te, va=None, iters=None, use_shift=True, use_day=True):
    """tr/te/va: lane-B rows only. yhat = P(copy)*z_sched + P(phys)*tau, shift rows get 86400+tau on top."""
    lab, keep, phys = _labels(tr)
    clf_cols = B_CAT + B_NUM + ["cross", "sched_hour"] + (DAY_COLS if use_day else [])
    vlab = vkeep = vphys = None
    if va is not None:
        vlab, vkeep, vphys = _labels(va)
    P, it_c = fit_clf(tr, lab, keep, te, va, vlab, vkeep, clf_cols, iters[0] if iters else None)
    tcols = TAU_CAT + tau_num(tr) + DAY + OCC + WX
    tau, it_t = fit_tau(tr, phys, te, va, vphys, tcols, iters[1] if iters else None)
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
    b = f[f["has_plan"] == 0]
    tr_all, ho = b[(b["split"] == "train") & ~b["month"].isin(HOLDOUT)], b[(b["split"] == "train") & b["month"].isin(HOLDOUT)]
    copy, near, phys, shift = classes(tr_all)
    print(pd.DataFrame({"apt": tr_all["apt"].values, "copy": copy, "near": near, "phys": phys, "shift": shift}).groupby("apt").sum())
    oof, its, t0 = np.full(len(tr_all), np.nan), [], time.time(); ll_day = ll_noday = n_ll = 0
    for i, months in enumerate(FOLDS):
        va = tr_all["month"].isin(months).values
        trn, van = tr_all[~va], tr_all[va]
        p, it, *_ = predict(trn, van, van)
        oof[va] = p; its.append(it)
        l1, n1 = clf_logloss(trn, van, True); l0, n0 = clf_logloss(trn, van, False)
        ll_day += l1 * n1; ll_noday += l0 * n0; n_ll += n1
        print(f"fold {i} {months}: rmse {rmse(van['y'].values, p):.1f} at {it} iters, {time.time() - t0:.0f}s", flush=True)
    print(f"day-feature multiclass logloss: with {ll_day / n_ll:.4f}, without {ll_noday / n_ll:.4f}")
    it = tuple(int(np.mean([x[j] for x in its]) * 1.1) for j in range(2))
    ph, _, P, rate, n_shift = predict(tr_all, ho, iters=it); print(f"shift rule: {APT_SHIFT} z_sched>{Z_SHIFT:.0f}, n={n_shift}, rate={rate:.3f}")
    dh = pd.read_parquet(CACHE / f"{donor}_holdout.parquet")[["id", "p"]].rename(columns={"p": "donor"})
    donor_p = ho[["id"]].merge(dh, on="id", how="left")["donor"].values; ph = blend_donor(ph, P[:, 0], donor_p)
    pd.DataFrame({"id": tr_all["id"].values, "oof": oof}).to_parquet(CACHE / "laneb_oof.parquet")
    pd.DataFrame({"id": ho["id"].values, "p": ph}).to_parquet(CACHE / "laneb_holdout.parquet")
    ho = ho.assign(p=ph); hb = ho.merge(dh, on="id")
    rmse_new, rmse_donor = rmse(hb["y"].values, hb["p"].values), rmse(hb["y"].values, hb["donor"].values)
    d = boot_delta(hb["y"].values, hb["donor"].values, hb["p"].values, hb["day"].values)
    abl = {"full": rmse_new, "donor_only": rmse_donor}
    for tag, kw in (("no_shift", dict(use_shift=False)), ("no_blend", {}), ("no_day", dict(use_day=False))):
        p, _, Pk, _, _ = predict(tr_all, ho, iters=it, **kw)
        p = blend_donor(p, Pk[:, 0], donor_p) if tag != "no_blend" else p
        abl[tag] = rmse(ho["y"].values, p)
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
        "# Lane B mechanism model\n\n",
        f"`PYTHONPATH=/c/Users/skf_s/prc26 python laneb.py cv {donor}` then `python laneb.py fit {it[0]} {it[1]}`.\n\n",
        f"Result: **{'PASS' if passed else 'FAIL'}** vs donor {donor} (1900.6). New {rmse_new:.1f}, donor {rmse_donor:.1f}, "
        f"bootstrap delta {d[0]:+.1f} [{d[1]:+.1f}, {d[2]:+.1f}]. Copy {copy_new:.1f} vs donor {copy_donor:.1f} ({'no worse' if copy_new <= copy_donor else 'worse'}).\n\n",
        f"Shift rule: {APT_SHIFT} z_sched > {Z_SHIFT:.0f}s, n={n_shift} train rows, empirical rate {rate:.3f}.\n\n",
        f"Day-feature multiclass log loss on CV folds: with {ll_d:.4f}, without {ll_nd:.4f} "
        f"({'day features help' if ll_d < ll_nd else 'day features do not help'}).\n\n",
        f"Donor blend: P(copy) > {DONOR_COPY_THRESH} rows take {DONOR_COPY_W:.2f}*ours + {1 - DONOR_COPY_W:.2f}*donor (tau misjudges the remainder there).\n\n",
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
    b = f[f["has_plan"] == 0]
    tr, rk = b[b["split"] == "train"], b[b["split"] == "rank"]
    p, _, P, _, _ = predict(tr, rk, iters=(it_clf, it_tau))
    dr = pd.read_parquet(f"data/{donor}_rank.parquet")[["MVT_ID_mvt", "TAXITIME_SEC_mvt"]].rename(columns={"MVT_ID_mvt": "id", "TAXITIME_SEC_mvt": "donor"})
    donor_p = rk[["id"]].merge(dr, on="id", how="left")["donor"].values; p = blend_donor(p, P[:, 0], donor_p)
    pd.DataFrame({"MVT_ID_mvt": rk["id"].values, "TAXITIME_SEC_mvt": p}).to_parquet("data/laneb_rank.parquet")
    print("data/laneb_rank.parquet", len(p), float(np.mean(p)))


if __name__ == "__main__":
    {"cv": lambda: cv(sys.argv[2]), "fit": lambda: fit(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4] if len(sys.argv) > 4 else "mix_cb_wx")}[sys.argv[1]]()
