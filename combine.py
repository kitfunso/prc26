"""Weighted sum of rank files: `python combine.py data/merry-octopus_v21.parquet data/mix_cb_wx_rank.parquet:0.648 data/mix_lgb_wxb_rank.parquet:0.352`."""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

ID, Y = "MVT_ID_mvt", "TAXITIME_SEC_mvt"


def main(out: str, parts: list[str]) -> None:
    acc = None
    for spec in parts:
        path, w = spec.rsplit(":", 1)
        d = pd.read_parquet(path).set_index(ID)[Y] * float(w)
        acc = d if acc is None else acc + d
    assert acc.notna().all(), "id sets differ between parts"
    pd.DataFrame({ID: acc.index, Y: np.maximum(acc.values, 0)}).to_parquet(out)
    print(out, len(acc), float(acc.mean()))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
