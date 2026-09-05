"""L6T: L6a's non-copy threshold sweep on lane-B CV (critique condition 3); a CV winner other than L6a's own 20 meets the holdout against L6a."""
from __future__ import annotations

import pandas as pd

from taxi.l5 import PSEUDO, judge
from taxi.l6 import MIN_NONCOPY, _mean, per_row

THRESHOLDS = (10, 15, 20, 25)
CELLS = {f"L6T{t}": t for t in THRESHOLDS}
K = len(THRESHOLDS)


def pred(tr: pd.DataFrame, te: pd.DataFrame, threshold: int) -> pd.Series:
    return per_row(tr, te, _mean, threshold)


def l6a(tr: pd.DataFrame, te: pd.DataFrame) -> pd.Series:
    return per_row(tr, te, _mean)


def main() -> None:
    judge(CELLS, pred, "L6T: lane-B gated mixture threshold sweep",
          f"Regenerate with `python -m taxi.l6t`. L6a's rule (P(copy) x z_sched + (1 - P(copy)) x the band's non-copy mean, P(copy) from the shrunk "
          f"airline x band table with {PSEUDO:.0f} pseudo-rows) with the band's non-copy minimum at {THRESHOLDS} training rows; L6a itself is "
          f"L6T{MIN_NONCOPY}, so that cell is the champion under another name and a CV tie keeps L6a.", "docs/L6T.md",
          champ=("L6a", l6a), k=K, same_as=f"L6T{MIN_NONCOPY}")


if __name__ == "__main__":
    main()
