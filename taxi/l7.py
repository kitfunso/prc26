"""L7: seasonal copy probability at the lane-B gate. LIRF's no-type copy rate is 0.718 in Nov-Apr and 0.474 in May-Oct on training months
(z = 7.9), so L6a's month-pooled P(copy) is wrong in both holdout months at once; only the probability changes, the mixture is L6a's."""
from __future__ import annotations

import numpy as np
import pandas as pd

from taxi.l4 import band
from taxi.l5 import NB, PSEUDO, Prob, is_copy, judge, shrunk
from taxi.l6 import CELLS as L6_CELLS, per_row

WINTER = (11, 12, 1, 2, 3, 4)
EPS = 1e-6


def _season(d: pd.DataFrame) -> np.ndarray:
    return d.month.isin(WINTER).values.astype(int)


def _shrink(num: np.ndarray, den: np.ndarray, prior: np.ndarray) -> np.ndarray:
    return (num + PSEUDO * prior) / (den + PSEUDO)


def _cell(keys_tr: list, keys_te: list, c: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """P(copy) for each te row from the training cell it falls in, shrunk toward prior with PSEUDO rows; an unseen cell is the prior."""
    g = pd.DataFrame({**{f"k{i}": k for i, k in enumerate(keys_tr)}, "c": c}).groupby([f"k{i}" for i in range(len(keys_tr))]).c.agg(["sum", "size"])
    s = g.reindex(pd.MultiIndex.from_arrays(keys_te))
    return _shrink(s["sum"].fillna(0).values, s["size"].fillna(0).values, prior)


def _band_prior(c: np.ndarray, tb: np.ndarray, zb: np.ndarray) -> np.ndarray:
    return pd.Series(c).groupby(tb).mean().reindex(range(NB)).fillna(c.mean()).values[zb]


def band_season(t: pd.DataFrame, te: pd.DataFrame) -> np.ndarray:
    """L7a: P(copy) per band and season, shrunk toward the band's own rate."""
    c, tb, zb = is_copy(t), band(t.z_sched.values), band(te.z_sched.values)
    return _cell([tb, _season(t)], [zb, _season(te)], c, _band_prior(c, tb, zb))


def airline_band_season(t: pd.DataFrame, te: pd.DataFrame) -> np.ndarray:
    """L7b: L5a's airline table with a season dimension, shrunk through band x season to band."""
    c, tb, zb = is_copy(t), band(t.z_sched.values), band(te.z_sched.values)
    s, zs = _season(t), _season(te)
    p_bs = _cell([tb, s], [zb, zs], c, _band_prior(c, tb, zb))
    return _cell([t.airline.values, tb, s], [te.airline.values, zb, zs], c, p_bs)


def season_odds(t: pd.DataFrame, te: pd.DataFrame) -> np.ndarray:
    """L7c: L5a's pooled table, its odds multiplied by the season's odds ratio against the pooled rate."""
    c, s, zs = is_copy(t), _season(t), _season(te)
    odds = lambda v: np.clip(v, EPS, 1 - EPS) / (1 - np.clip(v, EPS, 1 - EPS))
    base = odds(float(c.mean()))
    r = np.array([odds(float(c[s == k].mean())) / base if int((s == k).sum()) else 1.0 for k in (0, 1)])
    o = odds(shrunk(t, te)) * r[zs]
    return o / (1 + o)


CELLS: dict[str, Prob] = {"L7a": band_season, "L7b": airline_band_season, "L7c": season_odds}


def _pred(tr: pd.DataFrame, te: pd.DataFrame, prob: Prob) -> np.ndarray:
    return per_row(tr, te, L6_CELLS["L6a"], prob=prob)


def _champion(tr: pd.DataFrame, te: pd.DataFrame) -> np.ndarray:
    return per_row(tr, te, L6_CELLS["L6a"])


def main() -> None:
    judge(CELLS, _pred, "L7: lane-B seasonal copy probability",
          f"Regenerate with `python -m taxi.l7`. L6a's gated mixture throughout; only P(copy) changes. Season is Nov-Apr against May-Oct, set on "
          f"training months alone. L7a: band x season shrunk to the band rate. L7b: airline x band x season shrunk through band x season to the "
          f"band rate. L7c: L5a's table with its odds times the season odds ratio. Every table shrinks with {PSEUDO:.0f} pseudo-rows.",
          "docs/L7.md", champ=("L6a", _champion), k=3)


if __name__ == "__main__":
    main()
