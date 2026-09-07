"""LK1: the anchor never lands after takeoff, and no feature outside the declared gap family reproduces the label (S3 critique finding 6)."""
import os

import numpy as np
import pandas as pd
import pytest

from taxi.data import DATA, anchor, load
from taxi.lk1 import LIMIT, correlations


def test_anchor_takes_the_first_non_negative_gap_else_the_takeoff():
    df = pd.DataFrame({"has_fpl": [True, True, True, True, False, False],
                       "z_aobt": [100.0, -5.0, -5.0, np.nan, np.nan, np.nan],
                       "z_eobt": [200.0, 300.0, -1.0, np.nan, np.nan, np.nan],
                       "z_sched": [400.0, 500.0, 600.0, -9.0, 700.0, -3.0]})
    assert anchor(df).tolist() == [100.0, 300.0, 600.0, 0.0, 700.0, 0.0]


@pytest.fixture(scope="module")
def feb():
    if not os.path.isdir(DATA):
        pytest.skip("data/ not present")
    return load((2,))


def test_every_anchor_precedes_takeoff(feb):
    assert int((feb.z_anchor < 0).sum()) == 0


@pytest.mark.parametrize("lane", ["A", "B"])
def test_no_feature_outside_the_gap_family_reproduces_the_label(feb, lane):
    over = {c: round(r, 3) for c, r in correlations(feb[feb.has_fpl == (lane == "A")]).items() if r > LIMIT}
    assert not over, over


def test_the_ranking_frame_carries_no_label_column():
    """c = y - z_aobt, so any prediction-side use of it reads the row's own answer; the ranking file is the proof it is unavailable."""
    from taxi.data import LABELS, features
    if not os.path.exists("data/ranking.parquet"):
        pytest.skip("data/ranking.parquet not present")
    rk = features(["data/ranking.parquet"])
    assert not [c for c in LABELS if c in rk.columns]
    with pytest.raises(AttributeError):
        __import__("taxi.m2", fromlist=["is_tail"]).is_tail(rk)


def test_the_lane_a_file_path_runs_on_a_label_free_frame(feb, monkeypatch):
    """SB3 gated each predicted row's blend weight on is_tail(te) and so could not build the ranking file at all (H53)."""
    from taxi import sb3
    te = feb.head(300).drop(columns=["y", "c"])
    monkeypatch.setattr(sb3, "seed_pred", lambda s, tr, tea, *a, **k: np.full(len(tea), 900.0 + s))
    monkeypatch.setattr(sb3, "final_rounds", lambda *a, **k: (1, 1, 1))  # only the fits are stubbed: _fp is what broke second
    p = sb3.predict(feb.head(300), te, "ranking")
    assert len(p) == len(te) and np.isfinite(p).all()


def test_the_lane_b_file_path_runs_on_a_label_free_frame(feb, monkeypatch):
    """The other half of F36: the ranking file needs lane B too, and its gated mixture reads te through prob and band."""
    from taxi import l4, l6, l8
    from taxi.champ import CURRENT, predict_b
    from taxi.l5 import champion
    b = feb[~feb.has_fpl].reset_index(drop=True)
    tr, te = b.iloc[0::2].reset_index(drop=True), b.iloc[1::2].reset_index(drop=True).drop(columns=["y", "c"])
    monkeypatch.setattr(l4, "MIN_NOTYP", 5)  # February alone holds 58 no-type rows, so the real gate at 100 never opens on this slice
    monkeypatch.setitem(l8.CB_KW, "iterations", 5)
    p = l6.per_row(tr, te, l6.CELLS["L6a"], min_noncopy=1, prob=l8.CELLS[CURRENT["B"]])
    assert int((champion(tr, te) != p).sum()) > 0  # the mixture must actually fire, or this guards nothing
    assert len(p) == len(te) and np.isfinite(p).all()
    assert np.isfinite(predict_b(CURRENT["B"], tr, te)).all()  # and the composition the build calls
