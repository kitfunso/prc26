"""W1 weather lane (protocol section 2 register row, section 4 W1): METAR archive pull and the as-of join onto the feature frame."""
from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

STATIONS = ("EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM")
PULL = "2026-09-05"
RAW = Path("data/weather") / PULL
# IEM ASOS/METAR archive; report types 3 (routine hourly) and 4 (specials). One day before 2025 for the 3 h look-back.
URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
PARAMS = dict(data="all", year1=2024, month1=12, day1=31, year2=2026, month2=8, day2=2, tz="Etc/UTC",
              format="onlycomma", latlon="no", missing="M", trace="T", direct="no")
LOOKBACK = 3 * 3600.0
WX = ("TS", "SN", "FZ", "RA", "SH", "FG", "BR")
WEATHER = ("w_sknt", "w_gust", "w_vsby", "w_p01i", "w_tmpf", "w_ceil", "w_ts3h", "w_p3h", "w_age") + tuple(f"w_{c.lower()}" for c in WX)


def fetch() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / "manifest.json"
    prior = json.loads(path.read_text()) if path.exists() else {}
    manifest = {"url": URL, "params": PARAMS, "report_type": [3, 4], "fetched_utc": prior.get("fetched_utc", {}), "sha256": {}}
    for st in STATIONS:
        q = urllib.parse.urlencode({**PARAMS, "station": st}) + "&report_type=3&report_type=4"
        out = RAW / f"{st}.csv"
        if out.exists():
            print(f"{st}: exists, {out.stat().st_size:,} bytes (raw pulls are immutable; delete the directory to re-pull)")
        else:
            t0 = time.time()
            with urllib.request.urlopen(f"{URL}?{q}", timeout=600) as r:
                body = r.read()
            out.write_bytes(body)
            manifest["fetched_utc"][st] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            print(f"{st}: {len(body):,} bytes in {time.time() - t0:.0f} s, {body.count(b'\n') - 1:,} rows", flush=True)
        manifest["sha256"][out.name] = hashlib.sha256(out.read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest, indent=1))


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.replace({"M": np.nan, "T": 0.0}), errors="coerce")


def load_metar() -> pd.DataFrame:
    """One row per observation with numeric columns and present-weather flags, sorted by station and time."""
    frames = [pd.read_csv(RAW / f"{st}.csv", dtype=str, keep_default_na=False) for st in STATIONS]
    df = pd.concat(frames, ignore_index=True)
    out = pd.DataFrame({"apt": df.station, "t": pd.to_datetime(df.valid, utc=True).astype("int64") // 10**9})
    for c in ("sknt", "gust", "vsby", "p01i", "tmpf"):
        out[f"w_{c}"] = _num(df[c])
    out["w_gust"] = out.w_gust.fillna(0.0)
    layers = [np.where(df[f"skyc{i}"].isin(["BKN", "OVC", "VV"]), _num(df[f"skyl{i}"]), np.nan) for i in (1, 2, 3, 4)]
    out["w_ceil"] = np.nanmin(np.vstack(layers), axis=0) if len(df) else np.nan
    wx = df.wxcodes.fillna("")
    for c in WX:
        out[f"w_{c.lower()}"] = wx.str.contains(c).astype(np.int8)
    return out.sort_values(["apt", "t"]).reset_index(drop=True)


def join(df: pd.DataFrame, metar: pd.DataFrame) -> pd.DataFrame:
    """As-of join at the anchor time (latest observation at or before it) plus 3 h look-back sums, per airport."""
    anchor = df.t_off.values - df.z_anchor.values
    cols = {c: np.full(len(df), np.nan) for c in WEATHER}
    for apt, ix in df.groupby("apt").indices.items():
        m = metar[metar.apt == apt]
        if m.empty:
            continue
        t = m.t.values
        pos = np.searchsorted(t, anchor[ix], "right") - 1
        ok = pos >= 0
        p = np.where(ok, pos, 0)
        for c in ("w_sknt", "w_gust", "w_vsby", "w_p01i", "w_tmpf", "w_ceil", *[f"w_{c.lower()}" for c in WX]):
            cols[c][ix] = np.where(ok, m[c].values[p], np.nan)
        cols["w_age"][ix] = np.where(ok, anchor[ix] - t[p], np.nan)
        cum_ts, cum_p = np.cumsum(m.w_ts.values), np.cumsum(np.nan_to_num(m.w_p01i.values))
        lo = np.searchsorted(t, anchor[ix] - LOOKBACK, "left")
        hi = np.searchsorted(t, anchor[ix], "right")
        top, base = np.maximum(hi - 1, 0), np.maximum(lo - 1, 0)
        cols["w_ts3h"][ix] = np.where(hi > lo, cum_ts[top] - np.where(lo > 0, cum_ts[base], 0), 0)
        cols["w_p3h"][ix] = np.where(hi > lo, cum_p[top] - np.where(lo > 0, cum_p[base], 0), 0)
    out = df.copy()
    for c, v in cols.items():
        out[c] = v
    return out


if __name__ == "__main__":
    if sys.argv[1:2] == ["fetch"]:
        fetch()
    else:
        m = load_metar()
        print(m.groupby("apt").agg(n=("t", "size"), first=("t", "min"), last=("t", "max"), ts=("w_ts", "sum"), sn=("w_sn", "sum")).to_string())
        print(m[[*WEATHER[:6], "w_ts"]].describe().T.to_string())
