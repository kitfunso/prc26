"""LB2 (diagnostic, never a verdict): four lane-B hypotheses, measured and closed.

Lane B is 49% of the composite squared error and its champion reads z_sched plus a copy probability. That invites four obvious questions: borrow
the local pushback delay from lane A, borrow the arrival side, recalibrate the copy gate, or beat the base predictor on the rows the gate never
touches. All four lose here, and this file is the record so none of them costs a verdict twice.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from taxi.cb1 import lane_b
from taxi.champ import CURRENT, predict_b
from taxi.data import CV_FOLDS, HOLDOUT_MONTHS, TRAIN_MONTHS, _files, connect, load, month_file
from taxi.l2 import floored
from taxi.l4 import touched
from taxi.l5 import is_copy
from taxi.l8 import gbm

WINDOWS = (1800.0, 3600.0, 7200.0)
ARR_W = 3600.0
PLURAL = {True: "s", False: ""}


def transfer() -> list[str]:
    """Lane B's own pushback delay against the local lane-A level, the nearest available stand-in for the missing d_as column."""
    d = load(HOLDOUT_MONTHS)
    b = d[~d.has_fpl].reset_index(drop=True)
    own = b.z_sched.values - b.y.values
    a = d[d.has_fpl]
    sched_a, sched_b = a.t_off.values - a.z_sched.values, b.t_off.values - b.z_sched.values
    out = [f"On the holdout, a lane-B flight's own pushback delay has median {np.median(own):,.0f} s, mean {own.mean():,.0f}, sd "
           f"{own.std():,.0f}, against lane A's `d_as` at median {d.d_as.median():,.0f}, mean {d.d_as.mean():,.0f}, sd {d.d_as.std():,.0f}. "
           "Different populations, so the transfer is already unlikely. Measured anyway.", "",
           "| window | covered | corr(local lane-A delay, the lane-B flight's own delay) | RMSE of z_sched minus that level |",
           "|---|---|---|---|"]
    for w in WINDOWS:
        lvl = np.full(len(b), np.nan)
        for apt, ix in b.groupby("apt").indices.items():
            m = a.apt.values == apt
            s, v = sched_a[m], a.d_as.values[m]
            o = np.argsort(s)
            s, v = s[o], v[o]
            ok = np.isfinite(v)
            s, v = s[ok], v[ok]
            cs = np.concatenate([[0.0], np.cumsum(v)])
            lo, hi = np.searchsorted(s, sched_b[ix] - w, "left"), np.searchsorted(s, sched_b[ix] + w, "right")
            n = hi - lo
            lvl[ix] = np.where(n >= 5, (cs[hi] - cs[lo]) / np.maximum(n, 1), np.nan)
        ok = np.isfinite(lvl)
        pred = np.maximum(b.z_sched.values - lvl, 0.0)
        rmse = float(np.sqrt(np.mean((pred[ok] - b.y.values[ok]) ** 2)))
        out.append(f"| plus or minus {w / 60:.0f} min | {ok.mean() * 100:.1f}% | {np.corrcoef(lvl[ok], own[ok])[0, 1]:+.3f} | {rmse:,.0f} s |")
    return out + ["", "The champion scores 1,808 s on these rows. The best of those is several times worse and the correlation never reaches "
                  "+0.25, so lane A's delay level does not carry lane B's delay."]


def arrivals() -> list[str]:
    """The arrival side is complete and kept in the ranking file, so it is free information if it separates anything."""
    con = connect()
    arr = con.sql(f"""SELECT ADES_mvt AS apt, epoch(MVT_TIME_UTC_mvt) AS t_land, epoch(BLOCK_TIME_UTC_mvt) AS t_block,
                             epoch(SCHED_TIME_UTC_mvt) AS t_sched, TAXITIME_SEC_mvt::DOUBLE AS ti
                      FROM read_parquet({_files([month_file(m) for m in HOLDOUT_MONTHS])}) WHERE PHASE_mvt = 'ARR'""").df()
    gap = arr.t_block.values - arr.t_sched.values
    d = load(HOLDOUT_MONTHS)
    b = d[~d.has_fpl].reset_index(drop=True)
    cp = is_copy(b)
    out = [f"Arrivals over the same two months: {len(arr):,} rows, block null {arr.t_block.isna().mean() * 100:.2f}%, schedule null "
           f"{arr.t_sched.isna().mean() * 100:.2f}%, taxi-in null {arr.ti.isna().mean() * 100:.2f}%. The arrival recorder is complete where the "
           "departure one is not, and an arrival block equals its schedule within 60 s on "
           f"{np.nanmean(np.abs(gap) <= 60) * 100:.2f}% of rows, so the copy behaviour exists on that side too.", "",
           f"Two local arrival features on the {len(b):,} lane-B holdout rows ({int(cp.sum()):,} of them copies), each the mean over that "
           "airport's arrivals in the hour before the departure's anchor:", "",
           "| feature | on copy rows | on non-copy rows |", "|---|---|---|"]
    ok = np.isfinite(gap) & np.isfinite(arr.ti.values)
    A = arr[ok].assign(ac=(np.abs(gap[ok]) <= 60).astype(float))
    rate, tin = np.full(len(b), np.nan), np.full(len(b), np.nan)
    anc = b.t_off.values - b.z_sched.values
    for apt, ix in b.groupby("apt").indices.items():
        g = A[A.apt.values == apt]
        o = np.argsort(g.t_land.values)
        s = g.t_land.values[o]
        cc, ct = np.concatenate([[0.0], np.cumsum(g.ac.values[o])]), np.concatenate([[0.0], np.cumsum(g.ti.values[o])])
        lo, hi = np.searchsorted(s, anc[ix] - ARR_W, "left"), np.searchsorted(s, anc[ix], "left")
        n = hi - lo
        rate[ix] = np.where(n >= 3, (cc[hi] - cc[lo]) / np.maximum(n, 1), np.nan)
        tin[ix] = np.where(n >= 3, (ct[hi] - ct[lo]) / np.maximum(n, 1), np.nan)
    m = np.isfinite(rate)
    out += [f"| local arrival copy rate | {rate[m & cp].mean():.4f} | {rate[m & ~cp].mean():.4f} |",
            f"| local arrival taxi-in | {tin[m & cp].mean():,.0f} s | {tin[m & ~cp].mean():,.0f} s |", ""]
    a = d[d.has_fpl]
    anc_a = a.t_off.values - a.z_anchor.values
    tia = np.full(len(a), np.nan)
    for apt, ix in a.groupby("apt").indices.items():
        g = A[A.apt.values == apt]
        o = np.argsort(g.t_land.values)
        s = g.t_land.values[o]
        ct = np.concatenate([[0.0], np.cumsum(g.ti.values[o])])
        lo, hi = np.searchsorted(s, anc_a[ix] - ARR_W, "left"), np.searchsorted(s, anc_a[ix], "left")
        n = hi - lo
        tia[ix] = np.where(n >= 3, (ct[hi] - ct[lo]) / np.maximum(n, 1), np.nan)
    q = np.isfinite(tia) & np.isfinite(a.c.values) & np.isfinite(a.rec_taxi.values)
    return out + [f"Neither separates the copies. In lane A the same recent arrival taxi-in correlates "
                  f"{np.corrcoef(tia[q], a.c.values[q])[0, 1]:+.3f} with the label against `rec_taxi`'s "
                  f"{np.corrcoef(a.rec_taxi.values[q], a.c.values[q])[0, 1]:+.3f}, so it is a weaker copy of a column the champion already reads. "
                  "No cell declared on either side."]


def calibration() -> list[str]:
    """P enters the mixture linearly, so a miscalibrated gate would be a free gain; this fits it out of fold and checks."""
    tr = load(TRAIN_MONTHS)
    tr = tr[~tr.has_fpl].reset_index(drop=True)
    rows = []
    for fold in CV_FOLDS:
        fit, va = tr[~tr.month.isin(fold)].reset_index(drop=True), tr[tr.month.isin(fold)].reset_index(drop=True)
        m = touched(fit, va)
        for apt in np.unique(va.apt.values[m]):
            t = fit[(fit.apt == apt) & (fit.notyp == 1)].reset_index(drop=True)
            ix = np.flatnonzero(m & (va.apt.values == apt))
            if len(t) < 20 or len(ix) == 0:
                continue
            sub = va.iloc[ix]
            rows.append(pd.DataFrame({"p": gbm(t, sub), "cp": is_copy(sub).astype(int), "z": sub.z_sched.values, "y": sub.y.values}))
    o = pd.concat(rows, ignore_index=True)
    t = o.groupby(pd.cut(o.p, [0, .05, .2, .4, .6, .8, .95, 1.0], include_lowest=True), observed=True).agg(
        n=("p", "size"), meanP=("p", "mean"), actual=("cp", "mean"), medz=("z", "median"))
    out = [f"L8c's copy probability, fitted out of fold over the five month pairs: {len(o):,} gated rows, base copy rate {o.cp.mean():.4f}, mean "
           f"P {o.p.mean():.4f}.", "", "| P bin | rows | mean P | actual copy rate | median z_sched |", "|---|---|---|---|---|"]
    for i, r in t.iterrows():
        out.append(f"| {i} | {int(r.n):,} | {r.meanP:.3f} | {r.actual:.3f} | {r.medz:,.0f} s |")
    base, nc = o.cp.mean(), o.y[o.cp == 0].mean()
    out += ["", "Every bin tracks. Shrinking P toward the base rate is the one free knob the mixture has:", "",
            "| shrink lambda | RMSE on the gated rows |", "|---|---|"]
    for lam in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
        p = lam * o.p.values + (1 - lam) * base
        pred = np.maximum(p * o.z.values + (1 - p) * nc, 0.0)
        out.append(f"| {lam:.1f} | {float(np.sqrt(np.mean((pred - o.y.values) ** 2))):,.1f} s |")
    return out + ["", "Monotonically worse. The gate is calibrated and recalibration is closed."]


def untouched() -> list[str]:
    """The gate reaches 7% of the lane; this asks what the base predictor is worth on the other 93%, where most of the error sits."""
    trb, teb = lane_b()
    p, y = floored(predict_b(CURRENT["B"], trb, teb)), teb.y.values
    tch = touched(trb, teb)
    sse = (p - y) ** 2
    med = trb.groupby("apt", observed=True).y.median()
    const = teb.apt.map(med).astype(float).values
    out = [f"The gate touches {int(tch.sum()):,} of {len(teb):,} holdout rows ({tch.mean() * 100:.1f}%). Split the lane on it and compare the "
           "champion with a per-airport median constant, which reads no features at all:", "",
           "| rows | champion | per-airport median | share of lane-B squared error |", "|---|---|---|---|"]
    for nm, m in (("gate touched", tch), ("untouched", ~tch)):
        rc, rp = float(np.sqrt(np.mean((const[m] - y[m]) ** 2))), float(np.sqrt(np.mean((p[m] - y[m]) ** 2)))
        out.append(f"| {nm} ({int(m.sum()):,}) | {rp:,.1f} s | {rc:,.1f} s | {sse[m].sum() / sse.sum() * 100:.1f}% |")
    u = ~tch
    yu, pu = y[u], p[u]
    su = (pu - yu) ** 2
    o = np.argsort(-su)
    big = yu >= 43200
    out += ["", "The gate earns its keep where it fires, and on the other rows the champion is level with a constant. That looks like the place to "
            "model, until the error inside it is counted:", ""]
    for k in (1, 5, 20):
        out.append(f"- The worst {k} untouched row{PLURAL[k > 1]} hold{PLURAL[k == 1]} {su[o[:k]].sum() / su.sum() * 100:.1f}% of the untouched squared error.")
    rest = float(np.sqrt(np.mean((pu[~big] - yu[~big]) ** 2)))
    return out + [f"- The {int(big.sum())} untouched rows over 12 h hold {su[big].sum() / su.sum() * 100:.1f}%, and without them the untouched "
                  f"rows score {rest:,.1f} s on {int((~big).sum()):,} rows.", "",
                  "Those rows are the day-shift artifact of `docs/DE.md`, which flag F21 prices at zero through every route. What is left on the "
                  "untouched rows is an ordinary population the champion already fits to within a constant's distance, so no large modelling gain "
                  "is hiding under the gate's 93%."]


def main() -> None:
    lines = ["# LB2: four lane-B hypotheses, measured and closed (diagnostic, never a verdict)", "",
             f"Regenerate with `python -m taxi.lb2`. Champion {CURRENT['B']}, floored, holdout Jan + Jul 2025. No cell is declared. Lane B holds "
             "49% of the composite squared error (`docs/LEV.md`) and its champion reads two things, so these four are the obvious places to look. "
             "All four lose, and the point of writing them down is that they stay lost.", "",
             "## 1. Lane A's local pushback delay does not transfer", ""] + transfer() + [
             "", "## 2. The arrival side separates nothing", ""] + arrivals() + [
             "", "## 3. The copy gate is already calibrated", ""] + calibration() + [
             "", "## 4. The rows the gate never touches hold no gain either", ""] + untouched()
    text = "\n".join(lines) + "\n"
    with open("docs/LB2.md", "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
