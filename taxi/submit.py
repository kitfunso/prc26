"""Submission file builder: the template's ids, integer seconds, nothing missing (protocol section 8)."""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

TEMPLATE = "data/submitting.parquet"


def build_submission(pred: pd.Series, out: str, template: str = TEMPLATE) -> pd.DataFrame:
    """Write predictions indexed by MVT_ID_mvt as the organisers' parquet; raise on any id or null defect."""
    ids = duckdb.sql(f"SELECT MVT_ID_mvt FROM read_parquet('{template}')").df().MVT_ID_mvt
    missing, extra = set(ids) - set(pred.index), set(pred.index) - set(ids)
    if missing or extra:
        raise ValueError(f"id mismatch against the template: {len(missing)} missing, {len(extra)} extra")
    p = pred.reindex(ids.values)
    if p.isna().any():
        raise ValueError(f"{int(p.isna().sum())} null predictions")
    sub = pd.DataFrame({"MVT_ID_mvt": ids.values.astype("float64"),
                        "TAXITIME_SEC_mvt": np.rint(p.values).astype("int32")})
    duckdb.from_df(sub).write_parquet(out)
    return sub
