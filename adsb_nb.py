"""Second ADS-B layer: every departure on an ADS-B day reads its matched neighbours' first-layer corrections.

The first layer (adsb_stage.py) corrects departures with their own track. Its corrections, averaged over the matched
departures at the same airport within +-30 and +-60 min, say how far the airport runs from the model right now, which
also informs the departures with no track. Layer 2 is LightGBM on y - p1 over all departures on ADS-B days.

python adsb_nb.py gate [FLAGS]               nested cross-fit on the Jan+Jul 2025 holdout (layer-1 inputs to layer 2 never
                                             saw the test month), full-file paired day-block bootstrap against v53.
python adsb_nb.py apply SRC BASE OUT [FLAGS]  both layers trained on every labelled ADS-B day, applied to BASE.
FLAGS as adsb_stage.py (skip=X,Y, laneA).
"""
import glob, os, sys
import lightgbm as lgb
import numpy as np, pandas as pd
import adsb_stage as s1

ROOT, ADSB, APTS = s1.ROOT, s1.ADSB, s1.APTS
W = (1800, 3600)
P2 = dict(s1.PARAMS, num_leaves=15, min_data_in_leaf=500)
R2 = 400


def nb_cols(t, apt, c1, matched):
    """Mean first-layer correction and count of matched neighbours within each window, self excluded, plus all traffic."""
    res = {f"nb{w}": np.full(len(t), np.nan) for w in W} | {f"nbn{w}": np.zeros(len(t)) for w in W} | \
          {f"tot{w}": np.zeros(len(t)) for w in W}
    c = np.where(matched, c1, 0.0); n = matched.astype(float)
    for a in np.unique(apt):
        idx = np.where(apt == a)[0]
        o = idx[np.argsort(t[idx], kind="stable")]
        ts = t[o]
        cs, ns = np.concatenate([[0], np.cumsum(c[o])]), np.concatenate([[0], np.cumsum(n[o])])
        for w in W:
            lo, hi = np.searchsorted(ts, ts - w, "left"), np.searchsorted(ts, ts + w, "right")
            sm, k = cs[hi] - cs[lo] - c[o], ns[hi] - ns[lo] - n[o]
            res[f"nb{w}"][o] = np.where(k > 0, sm / np.maximum(k, 1), np.nan)
            res[f"nbn{w}"][o] = k
            res[f"tot{w}"][o] = hi - lo - 1
    return pd.DataFrame(res)


def frame2(d, c1, matched):
    x = pd.DataFrame({"p1": d.p.values + c1, "c1": c1, "matched": matched.astype(int), "has_plan": d.has_plan.values,
                      "apt": pd.Categorical(d.apt.values, categories=APTS)})
    return pd.concat([x, nb_cols(d.t.values, d.apt.values.astype(str), c1, matched)], axis=1)


def fit2(x, r):
    return lgb.train(P2, lgb.Dataset(x, r, categorical_feature=["apt"]), R2)


def layer1(m, x, tr_mask, te_mask):
    if not tr_mask.any() or not te_mask.any():
        return pd.Series(dtype=float)
    bst = s1.fit(x[tr_mask], (m.y - m.p).values[tr_mask])
    return pd.Series(bst.predict(x[te_mask]), index=m.id.values[te_mask])


def deps_2025():
    days = {os.path.basename(f)[:10] for f in glob.glob(os.path.join(s1.ADSB, "pts", "2025-*.parquet"))}
    b = pd.read_parquet(os.path.join(ADSB, "base_2025.parquet"))
    f = pd.read_parquet(os.path.join(ROOT, "data", "cache", "feat.parquet"), columns=["id", "mod", "z_lobt"])
    b = b[b.day.isin(days)].merge(f, on="id")
    b["t"] = pd.to_datetime(b.day).astype("int64") / 1e9 + b["mod"] * 60.0
    return b.reset_index(drop=True)


def corrections(d, c1s, flags):
    c1 = d.id.map(c1s).fillna(0.0).values
    matched = d.id.isin(c1s.index).values & s1.keep(d.apt.values, d.has_plan.values, flags)
    return np.where(matched, c1, 0.0), matched


def gate(*flags):
    sys.path.insert(0, ROOT)
    from train import boot_delta
    m, x = s1.train_rows()
    d = deps_2025()
    months = sorted(m.month.unique())
    out = []
    for T in (1, 7):
        S = [mo for mo in months if mo != T]
        c1s = pd.concat([layer1(m, x, m.month.isin([k for k in S if k != M]).values, (m.month == M).values) for M in S] +
                        [layer1(m, x, m.month.isin(S).values, (m.month == T).values)])
        c1, matched = corrections(d, c1s, flags)
        x2 = frame2(d, c1, matched)
        nb = x2.nbn3600.values > 0  # layer 2 only reads tracked neighbours, never recalibrates p alone
        tr, te = d.month.isin(S).values & nb, (d.month == T).values
        b2 = fit2(x2[tr], (d.y - d.p - c1).values[tr])
        c2 = np.where(s1.keep(d.apt.values[te], d.has_plan.values[te], flags) & nb[te], b2.predict(x2[te]), 0.0)
        out.append(pd.DataFrame({"id": d.id.values[te], "c1": c1[te], "c2": c2}))
        print(f"month {T}: layer 2 trained on {tr.sum()} rows, applied to {te.sum()}")
    o = pd.concat(out).set_index("id")
    b = pd.read_parquet(os.path.join(ADSB, "base_2025.parquet"))
    h = b[b.month.isin([1, 7])].reset_index(drop=True)
    z = h.id.map(pd.read_parquet(os.path.join(ROOT, "data", "cache", "feat.parquet"), columns=["id", "z_lobt"])
                  .set_index("id").z_lobt).values
    y, day, p = h.y.values, h.day.values, h.p.values
    c1, c2 = h.id.map(o.c1).fillna(0).values, h.id.map(o.c2).fillna(0).values
    rm = lambda q, k=slice(None): np.sqrt(((y[k] - q[k]) ** 2).mean())
    for name, q in (("clip only", s1.finish(p, z)), ("layer 1", s1.finish(p + c1, z)), ("layers 1+2", s1.finish(p + c1 + c2, z))):
        dd = boot_delta(y, p, q, day)
        print(f"{name:10s} holdout all {rm(p):.2f} -> {rm(q):.2f} | {dd[0]:+.3f} [{dd[1]:+.3f}, {dd[2]:+.3f}] | "
              f"SSE gain {(((y - p) ** 2).sum() - ((y - q) ** 2).sum()) / 1e6:+.1f}M")
    q1, q2 = s1.finish(p + c1, z), s1.finish(p + c1 + c2, z)
    for apt in APTS:
        k = h.apt.values == apt
        g1 = (((y[k] - q1[k]) ** 2).sum() - ((y[k] - q2[k]) ** 2).sum()) / 1e6
        print(f"  {apt} layer 2 over layer 1: SSE gain {g1:+.1f}M on {k.sum()} rows ({(c1[k] != 0).mean():.2f} matched)")
    tag = "_".join(("",) + flags)
    pd.DataFrame({"id": h.id, "p": q2}).to_parquet(os.path.join(ADSB, f"gate2_holdout{tag}.parquet"), index=False)


def apply(src, base, out, *flags):
    """Layer 1 out-of-fold on every labelled month feeds layer 2's training; both refit on everything for SRC."""
    m, x = s1.train_rows()
    d = deps_2025()
    months = sorted(m.month.unique())
    c1s = pd.concat([layer1(m, x, (m.month != M).values, (m.month == M).values) for M in months])
    c1, matched = corrections(d, c1s, flags)
    x2 = frame2(d, c1, matched)
    nb = x2.nbn3600.values > 0
    b2 = fit2(x2[nb], (d.y - d.p - c1).values[nb])
    bst = s1.fit(x, (m.y - m.p).values)
    ev = pd.read_parquet(os.path.join(ADSB, f"ev_{src}.parquet"))
    sub = pd.read_parquet(base)
    f = pd.read_parquet(os.path.join(ROOT, "data", "cache", "feat.parquet"), columns=["id", "apt", "has_plan", "z_lobt", "mod", "day"])
    r = sub.merge(f, left_on="MVT_ID_mvt", right_on="id")
    r = r[r.day.isin(set(ev.day))].reset_index(drop=True)  # layer 2 runs on ADS-B days only, as it was trained
    r["p"] = r.TAXITIME_SEC_mvt.values
    r["t"] = pd.to_datetime(r.day).astype("int64") / 1e9 + r["mod"] * 60.0
    e = r.merge(ev, on="MVT_ID_mvt")
    xe = s1.frame(e, e.p.values, e.apt.values, e.has_plan.values)
    c1s_new = pd.Series(bst.predict(xe), index=e.MVT_ID_mvt.values)
    c1n, mn = corrections(r, c1s_new, flags)
    xr = frame2(r, c1n, mn)
    c2 = np.where(s1.keep(r.apt.values, r.has_plan.values, flags) & (xr.nbn3600.values > 0), b2.predict(xr), 0.0)
    pn = s1.finish(r.p.values + c1n + c2, r.z_lobt.values)
    new = sub.set_index("MVT_ID_mvt").TAXITIME_SEC_mvt.copy()
    new.loc[r.MVT_ID_mvt.values] = pn
    o = sub.copy(); o["TAXITIME_SEC_mvt"] = sub.MVT_ID_mvt.map(new).astype(sub.TAXITIME_SEC_mvt.dtype).values
    assert len(o) == len(sub) and np.isfinite(o.TAXITIME_SEC_mvt).all() and (o.TAXITIME_SEC_mvt >= 0).all()
    o.to_parquet(out, index=False)
    mv = pn - r.p.values
    print(f"{out}: {len(r)} rows on ADS-B days, {mn.sum()} with their own track; mean move {mv.mean():+.1f} s, rms {np.sqrt((mv ** 2).mean()):.1f} s")


if __name__ == "__main__":
    {"gate": gate, "apply": apply}[sys.argv[1]](*sys.argv[2:])
