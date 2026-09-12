"""IEM ASOS METAR archive (data/weather/<pull>/<ICAO>.csv) and its as-of join at the anchor time."""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).parent / "data" / "weather" / "2026-09-05"
IEM = ("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station={st}&data=all&year1=2024&month1=12&day1=31"
       "&year2={y}&month2={m}&day2={d}&tz=Etc/UTC&format=onlycomma&latlon=no&elev=no&missing=M&trace=T&direct=no"
       "&report_type=3&report_type=4")
STATIONS = ("EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM")
LOOKBACK = 3 * 3600.0
CODES = ("TS", "SN", "FZ", "RA", "SH", "FG", "BR")
OBS = ("w_sknt", "w_gust", "w_vsby", "w_tmpf", "w_ceil") + tuple(f"w_{c.lower()}" for c in CODES)
WX = list(OBS) + ["w_ts3h", "w_age"]  # p01i is 0.00 on every European row, so no rain sums


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.replace({"M": np.nan, "T": 0.0}), errors="coerce")


def load_metar() -> pd.DataFrame:
    df = pd.concat([pd.read_csv(RAW / f"{st}.csv", dtype=str, keep_default_na=False) for st in STATIONS], ignore_index=True)
    out = pd.DataFrame({"apt": df["station"], "t": pd.to_datetime(df["valid"], utc=True).astype("int64") // 10**9})
    for c in ("sknt", "gust", "vsby", "tmpf"):
        out[f"w_{c}"] = _num(df[c])
    out["w_gust"] = out["w_gust"].fillna(0.0)
    layers = [np.where(df[f"skyc{i}"].isin(["BKN", "OVC", "VV"]), _num(df[f"skyl{i}"]), np.nan) for i in (1, 2, 3, 4)]
    with np.errstate(all="ignore"):
        out["w_ceil"] = np.nanmin(np.vstack(layers), axis=0)
    for c in CODES:
        out[f"w_{c.lower()}"] = df["wxcodes"].str.contains(c).astype(np.int8)
    return out.sort_values(["apt", "t"]).reset_index(drop=True)


def join(f: pd.DataFrame, metar: pd.DataFrame | None = None) -> pd.DataFrame:
    """Latest observation at or before the anchor (estimated off-block), plus 3 h look-back thunderstorm and rain sums."""
    metar = load_metar() if metar is None else metar
    at = f["t_off"].values - f["z_anchor"].values
    cols = {c: np.full(len(f), np.nan) for c in WX}
    for apt, ix in f.groupby("apt", observed=True).indices.items():
        m = metar[metar["apt"] == apt]
        if m.empty:
            continue
        t = m["t"].values
        pos = np.searchsorted(t, at[ix], "right") - 1
        ok, p = pos >= 0, np.maximum(pos, 0)
        for c in OBS:
            cols[c][ix] = np.where(ok, m[c].values[p], np.nan)
        cols["w_age"][ix] = np.where(ok, at[ix] - t[p], np.nan)
        lo, hi = np.searchsorted(t, at[ix] - LOOKBACK, "left"), np.searchsorted(t, at[ix], "right")
        cum = np.concatenate([[0.0], np.cumsum(m["w_ts"].values)])
        cols["w_ts3h"][ix] = cum[hi] - cum[lo]
    return pd.DataFrame(cols, index=f.index)


def fetch(pull: str) -> None:
    """`python weather.py fetch 2026-09-05`: one IEM ASOS CSV per station into data/weather/<pull>/."""
    y, m, d = pull.split("-")
    out = RAW.parent / pull
    out.mkdir(parents=True, exist_ok=True)
    for st in STATIONS:
        with urllib.request.urlopen(IEM.format(st=st, y=y, m=int(m), d=int(d)), timeout=300) as r:
            (out / f"{st}.csv").write_bytes(r.read())
        print(st, (out / f"{st}.csv").stat().st_size)


if __name__ == "__main__":
    if sys.argv[1:2] == ["fetch"]:
        fetch(sys.argv[2]); sys.exit()
    m = load_metar()
    print(m.groupby("apt").agg(n=("t", "size"), first=("t", "min"), last=("t", "max"), ts=("w_ts", "sum"), sn=("w_sn", "sum")).to_string())
