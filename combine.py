"""Weighted sum of rank files: `python combine.py data/merry-octopus_v21.parquet data/mix_cb_wx_rank.parquet:0.648 data/mix_lgb_wxb_rank.parquet:0.352`."""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

ID, Y = "MVT_ID_mvt", "TAXITIME_SEC_mvt"


def main(out: str, parts: list[str], over: str | None = None) -> None:
    acc = None
    for spec in parts:
        path, w = spec.rsplit(":", 1)
        d = pd.read_parquet(path).set_index(ID)[Y] * float(w)
        acc = d if acc is None else acc + d
    assert acc.notna().all(), "id sets differ between parts"
    if over:  # rows in `over`, by id, replace the weighted sum outright
        o = pd.read_parquet(over).set_index(ID)[Y]
        acc.loc[o.index] = o.values
    pd.DataFrame({ID: acc.index, Y: np.maximum(acc.values, 0)}).to_parquet(out)
    print(out, len(acc), float(acc.mean()))


if __name__ == "__main__":
    over = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--over=")), None)
    parts = [a for a in sys.argv[1:] if not a.startswith("--over=")]
    main(parts[0], parts[1:], over)
