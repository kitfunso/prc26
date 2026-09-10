"""PB claims a probe's gain is arithmetic, not an estimate: the SSE a measured group sum removes must be exact, overlaps and all."""
import numpy as np

from taxi.pb import removed


def test_the_removed_sse_matches_the_correction_it_predicts_on_overlapping_groups():
    rng = np.random.default_rng(0)
    n = 500
    w = np.column_stack([np.arange(n) < 300, np.arange(n) >= 300, rng.random(n) < 0.1]).astype(float)  # lane B straddles both months
    err = rng.normal(scale=50.0, size=n) + w @ np.array([12.0, -7.0, 250.0])
    c = w.T @ err
    d = w @ np.linalg.solve(w.T @ w, c)
    assert np.isclose(removed(w, c), float((err ** 2).sum() - ((err - d) ** 2).sum()))


def test_a_measured_group_sum_can_never_cost_sse():
    rng = np.random.default_rng(1)
    w = (np.arange(200) % 4 == 0).astype(float)[:, None]  # one group, no structure in it at all
    err = rng.normal(size=200)
    assert removed(w, w.T @ err) >= 0


def test_the_challenger_constant_names_our_best_scored_file():
    """PB rescales everything to CHALLENGER's MEASURED board result, so it must name the file the board actually ranks us on.

    This test asked for the NEWEST BUILT file until 2026-09-10, which held while every build beat the last one. v18, v19 and
    v20 are siblings off v17: the newest build is now neither the best nor necessarily scored at all, and pointing the carry at
    an unscored file would make `CHAL_BOARD` a projection, which is the one thing it says it is not.
    """
    import os

    from taxi.ceil import BEST, SCORED
    from taxi.pb import CHALLENGER

    assert os.path.basename(CHALLENGER) == f"merry-octopus_{BEST}.parquet", "update CHALLENGER and CHAL_BOARD in taxi/pb.py"
    for v in SCORED:
        if v.startswith("v"):
            assert os.path.exists(f"data/merry-octopus_{v}.parquet"), f"{v} carries a board score but no file was built"
