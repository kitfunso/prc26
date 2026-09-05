"""Brute-force check of the traffic-window counts on a sample of February departures."""
import numpy as np
import pytest

from taxi.data import LANDINGS, VIEW, WINDOW, _files, anchor, connect, month_file


@pytest.fixture(scope="module")
def raw():
    con, files = connect(), _files([month_file(2)])
    dep, arr = con.sql(VIEW.format(files=files)).df(), con.sql(LANDINGS.format(files=files)).df()
    dep["anchor"] = dep.t_off - anchor(dep)
    return dep, arr


def test_counts_match_brute_force(feb_features, raw):
    dep, arr = raw
    feats = feb_features.set_index("id")
    for i in np.random.default_rng(0).choice(len(dep), 300, replace=False):
        r = dep.iloc[i]
        same, land = dep[dep.apt == r.apt], arr[arr.apt == r.apt].t.values
        t, a, w = r.t_off, r.anchor, WINDOW
        expect = {
            "queue": int(((same.t_off >= a) & (same.t_off < t)).sum()),
            "dep_before": int(((same.t_off >= a - w) & (same.t_off < a)).sum()),
            "dep_after": int(((same.t_off > t) & (same.t_off <= t + w)).sum()),
            "arr_anchor": int(((land >= a - w) & (land <= a + w)).sum()),
            "arr_off": int(((land >= t - w) & (land <= t + w)).sum()),
            "push_anchor": int(((same.anchor >= a - w) & (same.anchor <= a + w)).sum()) - 1,
        }
        got = {k: int(v) for k, v in feats.loc[r.id, list(expect)].items()}
        assert got == expect, (r.id, got, expect)


def test_ranking_file_has_landings_at_every_airport():
    """The counts read the ARR rows of the scored file; a missing airport would silently zero them."""
    n = connect().sql(LANDINGS.format(files="['data/ranking.parquet']")).df().groupby("apt").size()
    assert len(n) == 10 and int(n.min()) > 10_000, n.to_dict()
