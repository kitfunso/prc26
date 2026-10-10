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
