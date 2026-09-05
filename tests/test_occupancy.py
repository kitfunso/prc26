"""Brute-force check of the stand-occupancy columns, the no-type flag and d_as on a sample of February departures."""
import numpy as np

from taxi.data import ONBLOCK, VIEW, _files, connect, month_file


def test_stand_columns_match_brute_force(feb_features):
    con, files = connect(), _files([month_file(2)])
    dep, arr = con.sql(VIEW.format(files=files)).df(), con.sql(ONBLOCK.format(files=files)).df()
    feats = feb_features.set_index("id")
    raw = dep.set_index("id").loc[feats.index]
    assert (feats.notyp.values == raw.notyp.values.astype(np.int8)).all()
    m = feats.z_aobt.notna().values
    assert np.allclose(feats.d_as.values[m], (feats.z_sched - feats.z_aobt).values[m])
    assert (feats.occ.values[raw.stand.isna().values] == 0).all()
    for r in dep[dep.stand.notna()].sample(300, random_state=0).itertuples():
        f = feats.loc[r.id]
        later = arr.t[(arr.apt == r.apt) & (arr.stand == r.stand) & (arr.t > r.t_off - f.z_anchor)]
        nxt = later.min() if len(later) else np.inf
        assert bool(f.occ) == (nxt < r.t_off), r.id
        if f.occ:
            assert abs(f.t_occ - (r.t_off - nxt)) < 1e-6, r.id
        else:
            assert np.isnan(f.t_occ), r.id
