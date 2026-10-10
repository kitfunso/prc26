"""Per-departure ADS-B ground events from the daily cuts of adsb_fetch.py.

python adsb_events.py SRC [FILE]   SRC train (2025 days), rank or final (FILE = the ranking parquet, default data/ranking.parquet);
writes data/adsb/ev_SRC.parquet, one row per matched departure keyed on MVT_ID_mvt. Each departure is matched to the aircraft
whose ground-to-air transition at the same airport is nearest its take-off (within MATCH_S), callsign and type as tie-breaks;
the track before that take-off gives candidate off-block clocks, every one a time before take-off in seconds.
"""
import glob, os, sys
from concurrent.futures import ProcessPoolExecutor
from datetime import timedelta
import numpy as np, pandas as pd

from adsb_fetch import APT, CODES, ELEV, LAT, LON, PTS, ROOT

MATCH_S, STILL_M, STOP_S, LONG_S = 120, 15.0, 60, 120
KX = {c: 111320.0 * np.cos(np.radians(APT[c][0])) for c in CODES}


def load(day):
    fs = [os.path.join(PTS, f"{d:%Y-%m-%d}.parquet") for d in (day - timedelta(1), day)]
    fs = [f for f in fs if os.path.exists(f)]
    p = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    for c in ("icao", "cs", "t"):
        p[c] = p[c].astype(str)
    la, lo = np.radians(p.lat.values.astype(float))[:, None], np.radians(p.lon.values.astype(float))[:, None]
    h = np.sin((np.radians(LAT) - la) / 2) ** 2 + np.cos(la) * np.cos(np.radians(LAT)) * np.sin((np.radians(LON) - lo) / 2) ** 2
    j = (12742.0 * np.arcsin(np.sqrt(h))).argmin(1)
    p["apt"] = np.array(CODES)[j]
    p["x"] = (p.lon.values - LON[j]) * np.array([KX[c] for c in CODES])[j]
    p["y"] = (p.lat.values - LAT[j]) * 110540.0
    p["relalt"] = p.alt.values - ELEV[j]
    return p.sort_values(["icao", "ts"], kind="stable").reset_index(drop=True)


def transitions(p):
    """Ground-to-air (take-off) and air-to-ground (landing) times per aircraft and airport."""
    g = p.ground.values
    same = (p.icao.values[1:] == p.icao.values[:-1]) & (p.apt.values[1:] == p.apt.values[:-1])
    dt = np.diff(p.ts.values)
    up = np.where(same & g[:-1] & ~g[1:] & (dt < 60))[0]
    dn = np.where(same & ~g[:-1] & g[1:] & (dt < 60))[0]
    to = pd.DataFrame({"i": up + 1, "icao": p.icao.values[up], "apt": p.apt.values[up], "t_air": p.ts.values[up + 1],
                       "gs_roll": p.gs.values[up], "cs": p.cs.values[up + 1], "t": p.t.values[up]})
    ld = pd.DataFrame({"icao": p.icao.values[dn], "apt": p.apt.values[dn], "t_land": p.ts.values[dn + 1]})
    return to, ld


def stops(ts, x, y):
    """Runs of STOP_S or more inside STILL_M of their first point: (start, end) times and the anchor index."""
    out, i, n = [], 0, len(ts)
    while i < n:
        j = i
        while j + 1 < n and (x[j + 1] - x[i]) ** 2 + (y[j + 1] - y[i]) ** 2 <= STILL_M ** 2:
            j += 1
        if ts[j] - ts[i] >= STOP_S:
            out.append((ts[i], ts[j], i))
        i = j + 1
    return out


def clocks(p, k, t_air, t_land):
    """Candidate off-block clocks, in seconds before t_air, from the aircraft's ground points ending at index k and going
    back to its last landing or a 30 min gap."""
    ts, x, y, gr, gs = p.ts.values, p.x.values, p.y.values, p.ground.values, p.gs.values
    lo = k
    while lo - 1 >= 0 and p.icao.values[lo - 1] == p.icao.values[k] and ts[k] - ts[lo - 1] < 6 * 3600 and \
            ts[lo] - ts[lo - 1] < 1800 and (np.isnan(t_land) or ts[lo - 1] >= t_land) and gr[lo - 1]:
        lo -= 1
    seg = slice(lo, k + 1)
    t, sx, sy = ts[seg], x[seg], y[seg]
    if len(t) < 2:
        return None
    # The point before the segment is often where the inbound flight parked before the transponder went off; how far
    # the first outbound point lies from it says how far the pushback had gone before the aircraft reappeared.
    pv = lo - 1
    prev = pv >= 0 and p.icao.values[pv] == p.icao.values[k] and p.apt.values[pv] == p.apt.values[k] and gr[pv] \
        and ts[lo] - ts[pv] < 12 * 3600
    d0 = np.hypot(sx - sx[0], sy - sy[0])
    moved = np.where(d0 > 2 * STILL_M)[0]
    t_move = t[moved[0]] if len(moved) else np.nan
    # Take-off roll start: the last point under 40 kt before the run-up; stops this close to it are runway holds.
    slow = np.where(np.nan_to_num(gs[seg], nan=0) < 40)[0]
    r0 = slow[-1] if len(slow) else len(t) - 1
    st = stops(t, sx, sy)
    far = [(a, b, ii) for a, b, ii in st if np.hypot(sx[ii] - sx[r0], sy[ii] - sy[r0]) > 400 and b - a >= LONG_S]
    lastfar = far[-1] if far else None
    path = np.hypot(np.diff(sx), np.diff(sy)).sum()
    gaps = np.diff(t)
    stop_tot = sum(b - a for a, b, _ in st if a >= (t_move if t_move == t_move else t[0]))
    lv = {f"lv{D}": t_air - t[w[0]] if len(w := np.where(d0 > D)[0]) else np.nan for D in (5, 10, 20)}
    # The longest stop is the parking dwell when the transponder stays on, but a de-icing pad or a hold otherwise,
    # so its leave time, length and distance from the take-off roll ride as separate columns.
    big = max(st, key=lambda s: s[1] - s[0]) if st else None
    if big:
        dp = np.hypot(sx - sx[big[2]], sy - sy[big[2]])
        w = np.where((dp > 10) & (t > big[1]))[0]
        lv.update(big_lv=t_air - t[w[0]] if len(w) else np.nan, big_len=big[1] - big[0], big_end=t_air - big[1],
                  big_droll=np.hypot(sx[big[2]] - sx[r0], sy[big[2]] - sy[r0]))
    return lv | {
        "first": t_air - t[0], "move": t_air - t_move, "dwell0": (t_move - t[0]) if t_move == t_move else t[-1] - t[0],
        "lastfar": (t_air - lastfar[1]) if lastfar else np.nan, "lastfar_len": (lastfar[1] - lastfar[0]) if lastfar else np.nan,
        "nstop": len(st), "stop_tot": stop_tot, "path_m": path, "gap_max": gaps.max(), "npts": len(t),
        "gs_first": gs[lo], "d_roll_first": np.hypot(sx[0] - sx[r0], sy[0] - sy[r0]),
        "land_gap": (t[0] - t_land) if t_land == t_land else np.nan, "seg_start_is_land": float(t_land == t_land and t[0] - t_land < 120),
        "tail_gap": t_air - t[-1], "x0": sx[0], "y0": sy[0],
        "prev_gap": ts[lo] - ts[pv] if prev else np.nan, "prev_d": np.hypot(x[lo] - x[pv], y[lo] - y[pv]) if prev else np.nan,
    }


def departures(src, path, day):
    lo = pd.Timestamp(day, tz="UTC"); hi = lo + pd.Timedelta(days=1)
    cols = ["MVT_ID_mvt", "ADEP_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "CALLSIGN_flt", "AIRCRAFT_TYPE_mvt", "STAND_mvt"]
    if src == "train":
        f = glob.glob(os.path.join(ROOT, "data", f"training_{day:%Y-%m}-01_*.parquet"))[0]
        d = pd.read_parquet(f, columns=cols)
    else:
        d = pd.read_parquet(path, columns=cols)
    d = d[(d.PHASE_mvt == "DEP") & (d.MVT_TIME_UTC_mvt >= lo) & (d.MVT_TIME_UTC_mvt < hi)].copy()
    d["mvt"] = d.MVT_TIME_UTC_mvt.astype("int64") / 1e6
    return d


def day_events(src, path, day):
    p = load(day)
    to, ld = transitions(p)
    d = departures(src, path, day)
    to = to[to.apt.isin(d.ADEP_mvt.unique())]
    m = d.merge(to, left_on="ADEP_mvt", right_on="apt", how="inner")
    m["dt_to"] = m.t_air - m.mvt
    m = m[m.dt_to.abs() <= MATCH_S].copy()
    m["cs_ok"] = (m.cs.str.strip() == m.CALLSIGN_flt.fillna("~")).astype(int)
    m["t_ok"] = (m.t == m.AIRCRAFT_TYPE_mvt.fillna("~")).astype(int)
    m["score"] = m.cs_ok * 1000 + m.t_ok * 300 - m.dt_to.abs()
    m = m.sort_values("score", ascending=False)
    m = m.drop_duplicates("MVT_ID_mvt").drop_duplicates(["icao", "t_air"])  # one track per departure and back
    n_cand = d.merge(to, left_on="ADEP_mvt", right_on="apt").assign(dt=lambda z: (z.t_air - z.mvt).abs()).query("dt <= @MATCH_S") \
        .groupby("MVT_ID_mvt").size()
    rows = []

    def land(icao, apt, t):
        lands = ld[(ld.icao == icao) & (ld.apt == apt) & (ld.t_land < t)].t_land
        return lands.max() if len(lands) else np.nan

    for r in m.itertuples(index=False):
        c = clocks(p, r.i - 1, r.t_air, land(r.icao, r.apt, r.t_air))
        if c is None:
            continue
        c.update(MVT_ID_mvt=r.MVT_ID_mvt, adep=r.ADEP_mvt, stand=r.STAND_mvt, mpass=1, dt_to=r.dt_to, cs_ok=r.cs_ok, t_ok=r.t_ok, gs_roll=r.gs_roll,
                 n_cand=int(n_cand.get(r.MVT_ID_mvt, 1)), icao=r.icao)
        rows.append(c)
    # Second pass: the take-off itself is often below the feeders, but the ground track under the departure's own
    # callsign is there, so the movement record's take-off time stands in for the transition.
    done = {c["MVT_ID_mvt"] for c in rows}
    um = d[~d.MVT_ID_mvt.isin(done) & d.CALLSIGN_flt.notna()]
    g = p[p.ground].assign(pos=lambda z: z.index)
    g = g.merge(um[["MVT_ID_mvt", "CALLSIGN_flt", "ADEP_mvt", "AIRCRAFT_TYPE_mvt", "STAND_mvt", "mvt"]], left_on=["cs", "apt"],
                right_on=["CALLSIGN_flt", "ADEP_mvt"])
    g = g[(g.ts >= g.mvt - 3600) & (g.ts <= g.mvt + 60)]
    last = g.sort_values("ts").drop_duplicates("MVT_ID_mvt", keep="last")
    for r in last.itertuples(index=False):
        c = clocks(p, r.pos, r.mvt, land(r.icao, r.apt, r.mvt))
        if c is None:
            continue
        c.update(MVT_ID_mvt=r.MVT_ID_mvt, adep=r.ADEP_mvt, stand=r.STAND_mvt, mpass=2, dt_to=np.nan, cs_ok=1, t_ok=int(r.t == r.AIRCRAFT_TYPE_mvt), gs_roll=r.gs, n_cand=1, icao=r.icao)
        rows.append(c)
    return pd.DataFrame(rows), len(d)


def main(src, path=None):
    path = path or os.path.join(ROOT, "data", "ranking.parquet")
    days = sorted(pd.Timestamp(os.path.basename(f)[:10]).date() for f in glob.glob(os.path.join(PTS, "*.parquet")))
    days = [d for d in days if (d.year == 2025) == (src == "train")]
    out, tot = [], 0
    with ProcessPoolExecutor(8) as ex:
        for day, (ev, n) in zip(days, ex.map(day_events, [src] * len(days), [path] * len(days), days)):
            print(f"{day} {len(ev)} of {n} departures matched", flush=True)
            out.append(ev.assign(day=str(day))); tot += n
    ev = pd.concat(out, ignore_index=True)
    ev.to_parquet(os.path.join(ROOT, "data", "adsb", f"ev_{src}.parquet"), index=False)
    print(f"{src}: {len(ev)} of {tot} departures matched over {len(days)} days")


if __name__ == "__main__":
    main(*sys.argv[1:3])
