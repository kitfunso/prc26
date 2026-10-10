"""A UTC month boundary cannot move labels out of the supplied-file scope."""
from __future__ import annotations

import pandas as pd

from campaign import outer_masks


def test_whole_month_file_excluded_when_utc_month_differs() -> None:
    rows = pd.DataFrame({"month": [8, 9, 1, 4], "file": ["training_2025-09-01_2025-10-01.parquet",
                                                         "training_2025-09-01_2025-10-01.parquet",
                                                         "training_2025-01-01_2025-02-01.parquet",
                                                         "training_2025-04-01_2025-05-01.parquet"]})
    fit, eva = outer_masks(rows, (9, 10))
    assert fit.tolist() == [False, False, False, True]
    assert eva.tolist() == [True, True, False, False]
