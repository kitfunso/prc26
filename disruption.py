"""Reporting-normalized weather and label-free recorder support for CP2."""
from __future__ import annotations

import numpy as np
import pandas as pd

from weather import load_metar


def _integral(t: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
    dt = np.minimum(np.r_[np.diff(t), 0], 3600)
    prefix = np.r_[0., np.cumsum(dt * values)]
    pos = np.searchsorted(t, query, "right") - 1
    ix = np.maximum(pos, 0)
    return np.where(pos >= 0, prefix[ix] + np.clip(query - t[ix], 0, 3600) * values[ix], 0)


def weather_features(f: pd.DataFrame) -> pd.DataFrame:
    metar = load_metar()
    out = pd.DataFrame(index=f.index)
    for col in ("d_w_coverage6", "d_w_snow6", "d_w_snow24", "d_w_freeze6", "d_w_lovis6"):
        out[col] = np.nan
    at = f.t_off.values - f.z_anchor.values
    for apt, ix in f.groupby("apt", observed=True).indices.items():
        m = metar[metar.apt == apt]
        if m.empty:
            continue
        t = m.t.values
        q = at[ix]
        for name, src, window in (("coverage6", None, 21600), ("snow6", "w_sn", 21600),
                                  ("snow24", "w_sn", 86400), ("freeze6", "w_fz", 21600),
                                  ("lovis6", "w_lovis", 21600)):
            v = np.ones(len(t)) if src is None else m[src].to_numpy(float)
            out.iloc[ix, out.columns.get_loc(f"d_w_{name}")] = (_integral(t, v, q) - _integral(t, v, q - window)) / window
    severity = out.d_w_snow6 + out.d_w_freeze6 + out.d_w_lovis6
    out["d_w_pressure"] = severity * f.queue.values
    out["d_w_push"] = severity * f.push_anchor.values
    out["d_w_stamp"] = severity * f.d_ae.abs().clip(upper=3600).values
    out["d_w_occ"] = severity * f.occ.values
    return out


def recorder_features(f: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=f.index)
    out["d_r_arr_support"] = np.log1p(f.w_n_arr) * (f.w_arr_a1 - f.w_arr_sc)
    out["d_r_day_support"] = np.log1p(f.ad_n_arr) * (f.ad_arr_a1 - f.ad_arr_sc)
    out["d_r_prev_support"] = (1 - f.pa_missing) * (f.pa_a1 - f.pa_sc)
    out["d_r_plus"] = f.wz_dae.clip(lower=0) * f.wz_dae_sd
    out["d_r_minus"] = f.wz_dae.clip(upper=0) * f.wz_dae_sd
    out["d_r_a_round"] = f.aobt_m5 * f.w_arr_sc
    out["d_r_neighbours"] = f.wz_n / f.n_day.clip(lower=1)
    out["d_r_occupancy"] = f.t_occ.clip(upper=7200) * f.w_arr_a1
    return out
