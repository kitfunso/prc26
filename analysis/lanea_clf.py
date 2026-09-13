"""Lane-A stamp-classifier headroom analysis (holdout months 1,7, has_plan==1).
`PYTHONPATH=/c/Users/skf_s/prc26 python analysis/lanea_clf.py oracle|confusion|dayclust|noisefloor`"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from data import CACHE, rmse
from features import build
from mix import Z, label
from train import HOLDOUT

pd.set_option("display.width", 250)


def _holdout_a() -> pd.DataFrame:
    f = build()
    ho = f[(f["split"] == "train") & f["month"].isin(HOLDOUT)]
    return ho[ho["has_plan"] == 1].copy()


def _mu(d: pd.DataFrame, p: np.ndarray, pa: np.ndarray, pe: np.ndarray, ps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Back out the physical estimate mu from the blend p and the three copy probs (mix.predict's yhat inverted)."""
    za, ze, zs = d["z_aobt"].values, d["z_eobt"].values, d["z_sched"].values
    den = 1 - pa - pe - ps
    ok = den > 0.02
    mu = np.full(len(d), np.nan)
    mu[ok] = (p[ok] - pa[ok] * np.nan_to_num(za[ok]) - pe[ok] * np.nan_to_num(ze[ok]) - ps[ok] * np.nan_to_num(zs[ok])) / den[ok]
    return mu, ok


def oracle() -> None:
    d = _holdout_a()
    lab = label(d)
    m = d.merge(pd.read_parquet(CACHE / "mix_cb_wx_holdout.parquet"), on="id")
    lab_m = label(m)  # re-derive on the merged frame order
    p, pa, pe, ps = m["p"].values, m["p_aobt"].values, m["p_eobt"].values, m["p_sched"].values
    y = m["y"].values
    actual = rmse(y, p)
    mu, ok = _mu(m, p, pa, pe, ps)
    truth = np.select([lab_m == 0, lab_m == 1, lab_m == 2], [m["z_aobt"].values, m["z_eobt"].values, m["z_sched"].values], default=np.nan)
    phys = lab_m == 3
    truth[phys] = np.where(ok[phys], mu[phys], p[phys])  # no-mu fallback: keep the model's own number
    n_fallback = int((phys & ~ok).sum())
    oracle_full = rmse(y, truth)

    aobt_only = p.copy(); aobt_only[lab_m == 0] = m["z_aobt"].values[lab_m == 0]
    eobt_only = p.copy(); eobt_only[lab_m == 1] = m["z_eobt"].values[lab_m == 1]

    print("=== mix_cb_wx, lane A holdout, n =", len(m), "===")
    print(f"actual RMSE:               {actual:.2f}")
    print(f"oracle classifier RMSE:    {oracle_full:.2f}  (mu fallback to actual p for {n_fallback} physical rows, denom<=0.02)")
    print(f"oracle AOBT-vs-rest only:  {rmse(y, aobt_only):.2f}")
    print(f"oracle EOBT-vs-rest only:  {rmse(y, eobt_only):.2f}")

    bl = d.merge(pd.read_parquet(CACHE / "blend_ra_cb+ra_cb_s1+ra_cb_s2+ra_lgb+nn3_holdout.parquet"), on="id")
    lab_b = label(bl)
    truth_b = bl["p"].values.copy()
    for k, z in enumerate(Z):
        m_k = lab_b == k
        truth_b[m_k] = bl[z].values[m_k]
    print("\n=== shipped blend (ra_cb+ra_cb_s1+ra_cb_s2+ra_lgb+nn3), lane A holdout, n =", len(bl), "===")
    print(f"actual RMSE:                          {rmse(bl['y'].values, bl['p'].values):.2f}")
    print(f"RMSE with exact copy-row substitution: {rmse(bl['y'].values, truth_b):.2f}")


def confusion() -> None:
    d = _holdout_a()
    m = d.merge(pd.read_parquet(CACHE / "mix_cb_wx_holdout.parquet"), on="id")
    lab = label(m)
    P = np.column_stack([m["p_aobt"].values, m["p_eobt"].values, m["p_sched"].values,
                         1 - m[["p_aobt", "p_eobt", "p_sched"]].values.sum(1)])
    pred = P.argmax(1)
    names = ["aobt", "eobt", "sched", "other"]
    ct = pd.crosstab(pd.Categorical(lab, categories=[0, 1, 2, 3]).rename_categories(names),
                      pd.Categorical(pred, categories=[0, 1, 2, 3]).rename_categories(names),
                      rownames=["true"], colnames=["pred"])
    print("=== confusion matrix (rows=true, cols=argmax pred) ===")
    print(ct.to_string())

    eps = 1e-12
    ll = -np.log(np.clip(P[np.arange(len(m)), lab], eps, 1)).mean()
    print(f"\nmulticlass log loss: {ll:.4f}")

    recall = pd.Series({names[k]: (pred[lab == k] == k).mean() for k in range(4)}).round(4)
    print("\nper-class recall:\n", recall.to_string())

    se = (m["y"].values - m["p"].values) ** 2
    tot = se.sum()
    cell = pd.DataFrame({"true": [names[k] for k in lab], "pred": [names[k] for k in pred], "se": se})
    share = cell.groupby(["true", "pred"], observed=True)["se"].sum().unstack(fill_value=0) / tot * 100
    print("\nsquared-error share per (true, pred) cell, % of total lane-A SSE:")
    print(share.round(2).to_string())


def _daysig(d: pd.DataFrame) -> pd.DataFrame:
    """Per-airport-day visible-stamp signature (no label/target use): shares and medians from AOBT/EOBT/LOBT/schedule fields."""
    d = d.assign(d_le=d["z_eobt"] - d["z_lobt"])
    g = d.groupby(["apt", "day"], observed=True)
    out = g.agg(n=("id", "size"), nB=("has_plan", lambda s: (s == 0).sum()),
                d_ae_med=("d_ae", "median"), aobt0_shr=("aobt_sec", lambda s: (s == 0).mean()),
                aobtm5_shr=("aobt_m5", "mean"), eobtm5_shr=("eobt_m5", "mean"),
                dae0_shr=("d_ae", lambda s: (s == 0).mean()), dle0_shr=("d_le", lambda s: (s == 0).mean()))
    out["bshare"] = out["nB"] / out["n"]
    return out.reset_index()


def dayclust() -> None:
    d = _holdout_a()
    lab = label(d)
    d = d.assign(cls=lab)
    for k, name in enumerate(("aobt", "eobt", "sched", "physical")):
        d[f"is_{name}"] = (d["cls"] == k).astype(int)

    g = d.groupby(["apt", "day"], observed=True)
    per_day = g.agg(n=("id", "size"), aobt_shr=("is_aobt", "mean"), eobt_shr=("is_eobt", "mean"), sched_shr=("is_sched", "mean"))
    apt_mean = d.groupby("apt", observed=True)[["is_aobt", "is_eobt", "is_sched"]].mean()
    apt_mean.columns = ["aobt_shr", "eobt_shr", "sched_shr"]
    per_day = per_day.join(apt_mean, on="apt", rsuffix="_apt")

    print("=== between-day dispersion vs binomial, per class (airport-days with n>=20) ===")
    pd20 = per_day[per_day["n"] >= 20]
    for cls in ("aobt", "eobt", "sched"):
        p_day, p_apt, n = pd20[f"{cls}_shr"].values, pd20[f"{cls}_shr_apt"].values, pd20["n"].values
        var_obs = np.average((p_day - p_apt) ** 2, weights=n)
        var_bin = np.average(p_apt * (1 - p_apt) / n, weights=n)
        print(f"{cls}: observed between-day var {var_obs:.5f}, binomial-expected var {var_bin:.5f}, dispersion ratio {var_obs / var_bin:.1f}x")

    per_day["hot"] = per_day[["eobt_shr", "sched_shr"]].max(1)
    top = per_day.sort_values("hot", ascending=False).head(15)
    sig = _daysig(d).set_index(["apt", "day"])
    top = top.join(sig[["nB", "bshare", "d_ae_med", "aobt0_shr", "aobtm5_shr", "eobtm5_shr", "dae0_shr", "dle0_shr"]], on=["apt", "day"])
    print("\n=== top 15 airport-days by max(eobt_shr, sched_shr) ===")
    print(top[["n", "aobt_shr", "eobt_shr", "sched_shr", "nB", "bshare", "d_ae_med", "aobt0_shr", "aobtm5_shr", "eobtm5_shr", "dae0_shr", "dle0_shr"]]
          .round(3).to_string())


def noisefloor() -> None:
    d = _holdout_a()
    lab = label(d)
    phys = d[lab == 3].copy()
    r = phys["y"].values - phys["z_aobt"].values
    q1, med, q3 = np.percentile(r, [25, 50, 75])
    share_600 = float((np.abs(r) > 600).mean())
    print(f"=== physical-class residual y - z_aobt, n={len(phys)} ===")
    print(f"median {med:.1f}, IQR [{q1:.1f}, {q3:.1f}], width {q3 - q1:.1f}, share |r|>600s: {share_600:.4f}")

    const = phys["z_aobt"].values + med
    print(f"best-constant (median shift) RMSE vs z_aobt: {rmse(phys['y'].values, const):.2f}  (z_aobt alone: {rmse(phys['y'].values, phys['z_aobt'].values):.2f})")

    g = phys.groupby(["apt", "hour"], observed=True)
    mu = g["y"].transform("mean")
    med_ah = g.apply(lambda x: (x["y"] - x["z_aobt"]).median(), include_groups=False)
    ah_pred = phys["z_aobt"].values + phys.groupby(["apt", "hour"], observed=True)["y"].transform(lambda s: (s - phys.loc[s.index, "z_aobt"]).median()).values
    print(f"airport-hour mean RMSE: {rmse(phys['y'].values, mu.values):.2f}")
    print(f"airport-hour median-residual + z_aobt RMSE: {rmse(phys['y'].values, ah_pred):.2f}")


if __name__ == "__main__":
    {"oracle": oracle, "confusion": confusion, "dayclust": dayclust, "noisefloor": noisefloor}[sys.argv[1]]()
