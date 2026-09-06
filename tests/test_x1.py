import numpy as np
import pytest
import pandas as pd

from taxi.x1 import levels_of, matrix
from taxi.bf import best_blend, needed_rho

COLS = ["apt", "runway", "z_aobt"]


def frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The shape taxi.data.features returns: apt an object column, runway a category, each frame carrying only its own levels."""
    fit = pd.DataFrame({"apt": ["EGLL", "EDDF", "EGLL"], "runway": pd.Categorical(["09R", "25L", "09R"]), "z_aobt": [1.0, 2.0, 3.0]})
    te = pd.DataFrame({"apt": ["EGLL", "LFPG"], "runway": pd.Categorical(["25L", "26"]), "z_aobt": [4.0, 5.0]})
    return fit, te


def test_predict_frame_takes_the_fit_frames_codes_and_drops_levels_it_never_saw():
    fit, te = frames()
    lv = levels_of(fit, COLS)
    a, b = matrix(fit, COLS, lv), matrix(te, COLS, lv)
    assert list(a.dtypes[["apt", "runway"]].map(str)) == ["category", "category"]
    assert list(b.apt.cat.categories) == list(a.apt.cat.categories)  # same codes mean the same airport in both frames
    assert b.apt.cat.codes.tolist() == [a.apt.cat.categories.get_loc("EGLL"), -1]  # LFPG is unseen, so it goes missing
    assert b.runway.isna().tolist() == [False, True]


def test_a_permuted_level_order_in_the_predict_frame_changes_nothing():
    fit, te = frames()
    lv = levels_of(fit, COLS)
    flipped = te.assign(runway=te.runway.cat.set_categories(["26", "25L"]))
    assert matrix(te, COLS, lv).runway.cat.codes.tolist() == matrix(flipped, COLS, lv).runway.cat.codes.tolist()


@pytest.mark.parametrize("bias", [0.0, 40.0])
def test_blend_frontier_matches_a_brute_force_weight_search(bias: float):
    """The bias case is the one that bites: RMSE is a second moment, so a centred correlation misprices the blend when errors are off zero."""
    rng = np.random.default_rng(0)
    y = rng.normal(size=4000)
    e0, ex = y + rng.normal(size=4000) + bias, 1.4 * y + rng.normal(size=4000) - bias / 3
    w, best, _ = best_blend(e0, ex)
    grid = [np.sqrt(np.mean((g * e0 + (1 - g) * ex) ** 2)) for g in np.linspace(-0.5, 1.5, 2001)]
    assert best <= min(grid) + 1e-12  # the closed form is exact, so no grid point can beat it
    assert abs(np.sqrt(np.mean((w * e0 + (1 - w) * ex) ** 2)) - best) < 1e-9


def test_needed_rho_is_the_correlation_at_which_the_gain_is_exactly_the_target():
    r0, rx, gain = 229.78, 240.0, 2.0
    rho = needed_rho(r0, rx, gain)
    at = lambda m: r0 - (lambda w: np.sqrt(w ** 2 * r0 ** 2 + (1 - w) ** 2 * rx ** 2 + 2 * w * (1 - w) * m * r0 * rx))(
        (rx ** 2 - m * r0 * rx) / (r0 ** 2 + rx ** 2 - 2 * m * r0 * rx))
    assert at(rho) >= gain > at(rho + 1e-3)  # lower correlation buys more, so the boundary sits just under the target
