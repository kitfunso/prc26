import numpy as np
import pandas as pd

from taxi.weather import join


def test_join_takes_latest_observation_at_or_before_anchor_and_sums_three_hours():
    metar = pd.DataFrame({
        "apt": ["LIRF"] * 3, "t": [1000.0, 2000.0, 3000.0],
        "w_sknt": [5.0, 10.0, 15.0], "w_gust": [0.0, 0.0, 20.0], "w_vsby": [6.0, 3.0, 1.0], "w_p01i": [0.0, 0.1, 0.2],
        "w_tmpf": [50.0, 51.0, 52.0], "w_ceil": [np.nan, 800.0, 500.0],
        "w_ts": [0, 1, 0], "w_sn": [0, 0, 0], "w_fz": [0, 0, 0], "w_ra": [0, 1, 1], "w_sh": [0, 0, 0], "w_fg": [0, 0, 0], "w_br": [0, 0, 1],
    })
    df = pd.DataFrame({"apt": ["LIRF", "LIRF", "LIRF", "EGLL"], "t_off": [2600.0, 2000.0, 900.0, 5000.0],
                       "z_anchor": [600.0, 0.0, 0.0, 0.0]})
    out = join(df, metar)
    assert out.w_sknt.tolist()[:3] == [10.0, 10.0, np.nan] or (out.w_sknt.iloc[0] == 10.0 and out.w_sknt.iloc[1] == 10.0 and np.isnan(out.w_sknt.iloc[2]))
    assert out.w_age.iloc[0] == 0.0 and out.w_age.iloc[1] == 0.0
    assert out.w_ts3h.iloc[0] == 1 and out.w_p3h.iloc[0] == 0.1
    assert out.w_ts3h.iloc[2] == 0 and out.w_p3h.iloc[2] == 0
    assert np.isnan(out.w_sknt.iloc[3]) and np.isnan(out.w_age.iloc[3])
    assert np.isnan(out.w_ceil.iloc[2]) and out.w_ceil.iloc[0] == 800.0
