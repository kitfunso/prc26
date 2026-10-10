"""ADS-B correction stage on top of a finished prediction file: LightGBM on the residual y - p of matched departures.

python adsb_stage.py gate             cross-fitted Jan+Jul 2025 holdout: each month corrected by a stage trained on the other
                                      months' ADS-B days (Feb and Jun out-of-fold, the other holdout month), full-file paired
                                      day-block bootstrap against v53's holdout.
python adsb_stage.py apply SRC BASE OUT   stage trained on every labelled ADS-B day, applied to ev_SRC rows of BASE (a submission
                                      parquet whose IDs match data/adsb/ev_SRC.parquet), written to OUT.
"""
import glob, os, sys
import lightgbm as lgb
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
ADSB = os.path.join(ROOT, "data", "adsb")
CLOCKS = ["first", "move", "lastfar", "lv5", "lv10", "lv20", "big_lv"]
RAW = ["dwell0", "lastfar_len", "nstop", "stop_tot", "path_m", "gap_max", "npts", "gs_first", "d_roll_first", "land_gap",
       "seg_start_is_land", "dt_to", "cs_ok", "t_ok", "n_cand", "gs_roll", "big_len", "big_end", "big_droll",
       "mpass", "tail_gap"]
PLAN = ["z_sched", "z_aobt", "z_eobt"]
APTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=31, min_data_in_leaf=200, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=0, num_threads=8)
ROUNDS = 600


OPT = {"apt": True, "clip": 0.0}
STANDS = None


def stand_map():
    """Each stand's position: the median first point of departures seen standing still there (60 s or more before the
    first move), pooled over every event file, labels unused."""
    ev = pd.concat([pd.read_parquet(f, columns=["adep", "stand", "x0", "y0", "dwell0"])
                    for f in glob.glob(os.path.join(ADSB, "ev_*.parquet"))])
    ev = ev[(ev.dwell0 >= 60) & ev.stand.notna()]
    g = ev.groupby(["adep", "stand"]).agg(sx=("x0", "median"), sy=("y0", "median"), sn=("x0", "size")).reset_index()
    return g[g.sn >= 5]


def frame(ev, p, apt, has_plan):
    x = pd.DataFrame({"p": p, "apt": pd.Categorical(apt, categories=APTS), "has_plan": has_plan})
    if not OPT["apt"]:
        x = x.drop(columns="apt")
    for c in CLOCKS:
        x[c] = ev[c].values
        x[f"{c}_d"] = ev[c].values - p
    for c in RAW:
        x[c] = ev[c].values
    sm = ev[["adep", "stand"]].merge(STANDS, on=["adep", "stand"], how="left")
    x["d_stand0"] = np.hypot(ev.x0.values - sm.sx.values, ev.y0.values - sm.sy.values)
    x["stand_n"] = sm.sn.values
    for c in PLAN:  # the label is a schedule copy on some rows, which no pushback clock can see
        x[c] = ev[c].values
        for k in ("lv10", "first"):
            x[f"{c}_{k}"] = ev[c].values - ev[k].values
    return x


def fit(x, r):
    if OPT["clip"]:
        r = np.clip(r, -OPT["clip"], OPT["clip"])
    cat = ["apt"] if "apt" in x else "auto"
    return lgb.train(PARAMS, lgb.Dataset(x, r, categorical_feature=cat), ROUNDS)


def train_rows():
    global STANDS
    STANDS = stand_map()
    ev = pd.read_parquet(os.path.join(ADSB, "ev_train.parquet"))
    b = pd.read_parquet(os.path.join(ADSB, "base_2025.parquet"))
    f = pd.read_parquet(os.path.join(ROOT, "data", "cache", "feat.parquet"), columns=["id"] + PLAN)
    m = ev.merge(b, left_on="MVT_ID_mvt", right_on="id").merge(f, on="id")
    return m, frame(m, m.p.values, m.apt.values, m.has_plan.values)


def gate(*flags):
    """Flags: noapt (no airport column), clipN (train on the residual clipped at +-N s), loao (each airport corrected by a
    stage that never saw it, the test for airports 2025 does not cover)."""
    sys.path.insert(0, ROOT)
    from train import boot_delta
    OPT["apt"] = "noapt" not in flags
    OPT["clip"] = next((float(f[4:]) for f in flags if f.startswith("clip")), 0.0)
    m, x = train_rows()
    b = pd.read_parquet(os.path.join(ADSB, "base_2025.parquet"))
    h = b[b.month.isin([1, 7])].reset_index(drop=True)
    corr = pd.Series(0.0, index=m.id.values)
    groups = [(mo, a) for mo in (1, 7) for a in APTS] if "loao" in flags else [(mo, None) for mo in (1, 7)]
    for mo, a in groups:
        tr, te = (m.month != mo).values, (m.month == mo).values
        if a:
            tr, te = tr & (m.apt != a).values, te & (m.apt == a).values
        if not te.any():
            continue
        bst = fit(x[tr], (m.y - m.p).values[tr])
        corr.loc[m.id.values[te]] = bst.predict(x[te])
        for s in separate(flags):
            ts, tt = tr & (m.apt == s).values, te & (m.apt == s).values
            if tt.any():
                corr.loc[m.id.values[tt]] = fit(x[ts], (m.y - m.p).values[ts]).predict(x[tt]) if ts.any() else 0.0
        print(f"month {mo} {a or ''}: trained on {tr.sum()} rows, corrected {te.sum()}")
    c = h.id.map(corr).fillna(0.0).values
    c = np.where(keep(h.apt.values, h.has_plan.values, flags), c, 0.0)
    z = h.id.map(pd.read_parquet(os.path.join(ROOT, "data", "cache", "feat.parquet"), columns=["id", "z_lobt"])
                  .set_index("id").z_lobt).values
    pn = finish(h.p.values + c, z)
    y, day = h.y.values, h.day.values
    rm = lambda q, k=slice(None): np.sqrt(((y[k] - q[k]) ** 2).mean())
    a = h.has_plan.values == 1
    print(f"holdout all {rm(h.p.values):.2f} -> {rm(pn):.2f} | lane A {rm(h.p.values, a):.2f} -> {rm(pn, a):.2f} | "
          f"lane B {rm(h.p.values, ~a):.2f} -> {rm(pn, ~a):.2f} | corrected rows {(c != 0).sum()}")
    d = boot_delta(y, h.p.values, pn, day)
    print(f"paired day-block bootstrap, all rows {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}] | SSE gain "
          f"{(((y - h.p.values) ** 2).sum() - ((y - pn) ** 2).sum()) / 1e6:+.1f}M")
    hit = h.id.isin(m.id).values
    for apt in APTS:
        for lane, lk in (("A", a), ("B", ~a)):
            k = hit & (h.apt.values == apt) & lk
            if k.sum():
                g = (((y[k] - h.p.values[k]) ** 2).sum() - ((y[k] - pn[k]) ** 2).sum()) / 1e6
                print(f"  {apt} {lane} n={k.sum()} {rm(h.p.values, k):.1f} -> {rm(pn, k):.1f} SSE gain {g:+.1f}M")
    tag = "_".join(("",) + flags)
    pd.DataFrame({"id": h.id, "p": pn}).to_parquet(os.path.join(ADSB, f"gate_holdout{tag}.parquet"), index=False)


def tune(*sets):
    """Each SET (k=v,k=v; rounds=N allowed) fits on the Jan+Jul 2025 days and is scored on every other labelled month,
    so settings are chosen away from the holdout the gate reads."""
    global ROUNDS
    m, x = train_rows()
    tr = m.month.isin([1, 7]).values
    va = ~tr & ~m.apt.isin(["LFPG", "LIRF"]).values  # scored where the stage is applied
    y, p = m.y.values[va], m.p.values[va]
    base = dict(PARAMS), ROUNDS
    for s in ("",) + sets:
        kv = dict(t.split("=") for t in s.split(",") if t)
        ROUNDS = int(kv.pop("rounds", base[1]))
        PARAMS.clear(); PARAMS.update(base[0], **{k: type(base[0].get(k, 1.0))(v) for k, v in kv.items()})
        q = p + fit(x[tr], (m.y - m.p).values[tr]).predict(x[va])
        print(f"{s or 'default':40s} rounds {ROUNDS}: matched RMSE {np.sqrt(((y - p) ** 2).mean()):.2f} -> "
              f"{np.sqrt(((y - q) ** 2).mean()):.2f} on {va.sum()} rows, months {sorted(set(m.month[va]))}")
    PARAMS.clear(); PARAMS.update(base[0]); ROUNDS = base[1]


def separate(flags):
    """sep=X,Y: those airports get a stage fitted on their own rows only, where the pooled one learns the wrong lesson
    (LIRF labels often copy the schedule, which the pushback clock never sees)."""
    return next((f[4:].split(",") for f in flags if f.startswith("sep=")), [])


def keep(apt, has_plan, flags):
    """laneA: lane A rows only; skip=X,Y: leave those airports uncorrected."""
    skip = next((f[5:].split(",") for f in flags if f.startswith("skip=")), [])
    k = ~np.isin(apt, skip)
    return k & (has_plan == 1) if "laneA" in flags else k


def finish(p, z):
    p = np.where(np.isnan(z), p, np.clip(p, z - 3606, z + 3606))  # |BLOCK - LOBT| <= 3606 s on every 2025 departure
    return np.maximum(p, 0)


def apply(src, base, out, *flags):
    OPT["apt"] = "noapt" not in flags
    OPT["clip"] = next((float(f[4:]) for f in flags if f.startswith("clip")), 0.0)
    m, x = train_rows()
    bst = fit(x, (m.y - m.p).values)
    ev = pd.read_parquet(os.path.join(ADSB, f"ev_{src}.parquet"))
    s = pd.read_parquet(base)
    feat = pd.read_parquet(os.path.join(ROOT, "data", "cache", "feat.parquet"), columns=["id", "apt", "has_plan", "z_lobt"] + PLAN)
    e = ev.merge(s, on="MVT_ID_mvt").merge(feat, left_on="MVT_ID_mvt", right_on="id")
    xe = frame(e, e.TAXITIME_SEC_mvt.values, e.apt.values, e.has_plan.values)
    c = bst.predict(xe)
    for s in separate(flags):
        k = e.apt.values == s
        if k.any():
            c[k] = fit(x[(m.apt == s).values], (m.y - m.p).values[(m.apt == s).values]).predict(xe[k])
    c = np.where(keep(e.apt.values, e.has_plan.values, flags), c, 0.0)
    pn = finish(e.TAXITIME_SEC_mvt.values + c, e.z_lobt.values)
    new = s.set_index("MVT_ID_mvt").TAXITIME_SEC_mvt.copy()
    new.loc[e.MVT_ID_mvt.values] = pn
    o = s.copy(); o["TAXITIME_SEC_mvt"] = s.MVT_ID_mvt.map(new).astype(s.TAXITIME_SEC_mvt.dtype).values
    assert len(o) == len(s) and np.isfinite(o.TAXITIME_SEC_mvt).all() and (o.TAXITIME_SEC_mvt >= 0).all()
    o.to_parquet(out, index=False)
    print(f"{out}: {len(e)} rows corrected, mean move {(pn - e.TAXITIME_SEC_mvt.values).mean():+.1f} s, "
          f"rms move {np.sqrt(((pn - e.TAXITIME_SEC_mvt.values) ** 2).mean()):.1f} s")


if __name__ == "__main__":
    {"gate": gate, "apply": apply, "tune": tune}[sys.argv[1]](*sys.argv[2:])
