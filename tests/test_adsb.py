"""ADS-B stage units: the byte prefilter never drops a point the cut keeps, the clock and stop readers, the output clip."""
from __future__ import annotations

import numpy as np
import pandas as pd

import adsb_events as E
import adsb_fetch as F
import adsb_stage as S


def _point(lat: float, lon: float) -> bytes:
    return f'"trace":[ \n[12.34,{lat:.6f},{lon:.6f},"ground",3.1,90.0,0,0,null,"adsb_icao"]'.encode()


def test_prefilter_keeps_every_point_inside_the_radius():
    rng = np.random.default_rng(0)
    for la, lo in zip(F.LAT, F.LON):
        r = F.RADIUS_KM * np.sqrt(rng.random(400))
        a = rng.random(400) * 2 * np.pi
        lats = la + r * np.sin(a) / 110.574
        lons = lo + r * np.cos(a) / (111.320 * np.cos(np.radians(la)))
        assert all(F.NEAR.search(_point(x, y)) for x, y in zip(lats, lons))


def test_prefilter_rejects_points_far_from_every_airport():
    for lat, lon in ((0.0, 0.0), (27.3, -82.5), (50.03, 9.5), (45.0, 8.55), (40.49, 3.57)):
        assert not F.NEAR.search(_point(lat, lon))


def test_tenths_truncate_toward_zero_like_json():
    assert F._tenths(-3.716, -3.416) == ["-3.4", "-3.5", "-3.6", "-3.7"]
    assert F._tenths(49.93, 50.13) == ["49.9", "50.0", "50.1"]


def test_finish_clips_to_the_lobt_window_and_floors_at_zero():
    p, z = np.array([10000.0, -50.0, 500.0]), np.array([1000.0, 300.0, np.nan])
    assert np.allclose(S.finish(p, z), [1000.0 + 3606, 0.0, 500.0])


def test_flags_parse():
    assert S.separate(("skip=LIRF", "sep=LIRF,EGLL")) == ["LIRF", "EGLL"]
    k = S.keep(np.array(["LIRF", "EDDF", "EDDF"]), np.array([1, 1, 0]), ("skip=LIRF", "laneA"))
    assert k.tolist() == [False, True, False]
    k = S.keep(np.array(["LIRF", "LIRF", "EDDF"]), np.array([1, 0, 0]), ("skipB=LIRF",))
    assert k.tolist() == [True, False, True]


def test_copy_rates_leave_the_rows_own_month_out():
    src = pd.DataFrame({"adep": "EDDF", "stand": "A1", "operator": "op", "month": [1, 2], "y": [1000.0, 1000.0],
                        "z_eobt": [1000.0, 2000.0], "z_sched": [5000.0, 3000.0], "z_lobt": [np.nan, np.nan],
                        "lv10": [500.0, 1000.0]})
    assert S.copy_flags(src)[""].tolist() == [True, False]
    lomo = S.copy_rates(src, src, True)["cr_stand"]
    assert np.allclose(lomo, [(0 + 25) / 51, (1 + 25) / 51])  # prior 0.5 x 50 rows; each row sees only the other month
    full = S.copy_rates(src, src.iloc[:1], False)["cr_operator"]
    assert np.allclose(full, [(1 + 25) / 52])
    try:
        S.set_opts(("cr3",))
        r = S.copy_rates(src, src.iloc[:1], False)
        assert S.OPT["cr"] and np.allclose(r["cr_stand_eobt"], [(1 + 25) / 52]) and np.allclose(r["cr_stand_sched"], [0.0])
        S.set_opts(("crh",))
        r = S.copy_rates(src.assign(hour=[7, 9]), src.iloc[:1].assign(hour=7), False)
        assert S.OPT["cr"] and "hour" in S.plan_cols() and np.allclose(r["cr_hour"], [(1 + 25) / 51])
    finally:
        S.set_opts(())


def test_clock_bias_skips_stamp_copies_and_missing_clocks():
    src = pd.DataFrame({"adep": "EDDF", "stand": "A1", "month": [1, 2, 3], "y": [1000.0, 900.0, 800.0],
                        "z_eobt": [1000.0, 5000.0, 5000.0], "z_sched": 5000.0, "z_lobt": np.nan,
                        "lv10": [500.0, 600.0, np.nan], "first": [400.0, 500.0, 300.0]})
    cb = S.clock_bias(src, src.iloc[:1], False)  # row 0 copies its EOBT, so only rows 1 and 2 count
    assert np.allclose(cb["cb_lv10"], [300.0]) and np.allclose(cb["cb_first"], [450.0]) and len(cb) == 2
    try:
        S.set_opts(("cbo",))
        cb = S.clock_bias(src.assign(operator=["a", "b", "b"]), src.iloc[:1].assign(operator="b"), False)
        assert S.OPT["cb"] and np.allclose(cb["cb_first_operator"], [450.0]) and np.allclose(cb["cb_lv10"], [300.0])
    finally:
        S.set_opts(())


def test_rec_adds_the_other_stamps_and_the_recording_columns():
    cols = S.CLOCKS + S.RAW + S.PLAN + S.STAMPS + S.REC + ["adep", "stand", "x0", "y0"]
    ev = pd.DataFrame({c: [1.0, 2.0] for c in cols}).assign(adep="EDDF", stand="A1", z_lobt=[500.0, 600.0])
    S.STANDS = pd.DataFrame({"adep": ["EDDF"], "stand": ["A1"], "sx": [0.0], "sy": [0.0], "sn": [9]})
    try:
        S.set_opts(("rec",))
        x = S.frame(ev, np.zeros(2), np.array(["EDDF", "EDDF"]), np.ones(2))
        assert x["z_lobt_lv10"].tolist() == [499.0, 598.0] and set(S.REC) <= set(x.columns)
        S.set_opts(())
        assert not {"z_lobt", "z_iobt_first", "b_share"} & set(S.frame(ev, np.zeros(2), np.array(["EDDF"] * 2), np.ones(2)))
    finally:
        S.set_opts(())


def test_clocks_read_the_stand_dwell_and_the_leave_time():
    ts = np.arange(0.0, 1200.0, 10.0)  # 10 min parked, then 1 m/s along x
    x = np.where(ts < 600, 0.0, ts - 600)
    p = pd.DataFrame({"ts": ts, "x": x, "y": 0.0, "ground": True, "gs": np.where(ts < 600, 0.0, 2.0),
                      "icao": "abc123", "apt": "EDDM"})
    t_air = 1300.0
    c = E.clocks(p, len(p) - 1, t_air, np.nan)
    assert c["first"] == t_air and c["dwell0"] == 640.0  # the first point more than 30 m out is at 640 s
    assert c["lv10"] == t_air - 620.0 and c["nstop"] == 1
    st = E.stops(ts, x, np.zeros_like(x))
    assert st[0][0] == 0.0 and st[0][1] >= 590.0
