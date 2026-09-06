"""L6: L5a's airline mixture only in a band with enough non-copy training rows (the H26 loss sat in the far band, whose non-copy rows are not taxi times)."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from taxi.l4 import MIN_BAND, band, touched
from taxi.l3 import COPY
from taxi.l5 import NB, PSEUDO, Prob, champion, is_copy, judge, shrunk

MIN_NONCOPY = MIN_BAND  # inherited from L4's band minimum, not tuned: LIRF's non-copy counts gate the same bands for any threshold in 12..35
Fit = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def _mean(z_tr: np.ndarray, y_tr: np.ndarray, z_te: np.ndarray) -> np.ndarray:
    return np.full(len(z_te), float(y_tr.mean()))


def _line(z_tr: np.ndarray, y_tr: np.ndarray, z_te: np.ndarray) -> np.ndarray:
    a, s = np.linalg.lstsq(np.column_stack([np.ones(len(z_tr)), z_tr]), y_tr, rcond=None)[0]
    return a + s * z_te


def per_row(tr: pd.DataFrame, te: pd.DataFrame, fit: Fit, min_noncopy: int = MIN_NONCOPY, prob: Prob = shrunk,
            copy_s: float = COPY) -> np.ndarray:
    """L4b everywhere; a touched row whose band has min_noncopy non-copy training rows gets P(copy) x z_sched + (1 - P(copy)) x fit(non-copy rows)."""
    out, m = champion(tr, te), touched(tr, te)
    for apt in np.unique(te.apt.values[m]):
        t = tr[(tr.apt == apt) & (tr.notyp == 1)].reset_index(drop=True)
        ix = np.flatnonzero(m & (te.apt.values == apt))
        sub, c, tb = te.iloc[ix], is_copy(t, copy_s), band(t.z_sched.values)
        z, zb, p = sub.z_sched.values, band(sub.z_sched.values), prob(t, sub)
        for b in np.unique(zb):
            k, j = (tb == b) & ~c, zb == b
            if int(k.sum()) >= min_noncopy:
                out[ix[j]] = p[j] * z[j] + (1 - p[j]) * fit(t.z_sched.values[k], t.y.values[k], z[j])
    return out


def gate_counts(tr: pd.DataFrame, te: pd.DataFrame) -> dict[str, list[int]]:
    """Non-copy training rows per band at each touched airport; the mixture applies where the count reaches MIN_NONCOPY."""
    out, m = {}, touched(tr, te)
    for apt in np.unique(te.apt.values[m]):
        t = tr[(tr.apt == apt) & (tr.notyp == 1)]
        c, tb = is_copy(t), band(t.z_sched.values)
        out[str(apt)] = [int(((tb == b) & ~c).sum()) for b in range(NB)]
    return out


CELLS: dict[str, Fit] = {"L6a": _mean, "L6b": _line}


def main() -> None:
    judge(CELLS, per_row, "L6: lane-B gated airline mixture",
          f"Regenerate with `python -m taxi.l6`. A touched row is re-predicted only when its band has at least {MIN_NONCOPY} non-copy training rows at "
          f"the airport: P(copy) x z_sched + (1 - P(copy)) x the band's non-copy component, P(copy) from L5a's shrunk airline x band table "
          f"({PSEUDO:.0f} pseudo-rows). L6a: the component is the band's non-copy mean. L6b: least squares y on [1, z_sched] over the band's non-copy "
          f"rows.", "docs/L6.md")


if __name__ == "__main__":
    main()
