"""Submission file builder: the template's ids, integer seconds, nothing missing (protocol section 8)."""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

TEMPLATE = "data/submitting.parquet"


def build_submission(pred: pd.Series, out: str, template: str = TEMPLATE, floor: float | None = None) -> pd.DataFrame:
    """Write predictions indexed by MVT_ID_mvt as the organisers' parquet; raise on any id or null defect; floor clips (C1, off by default)."""
    ids = duckdb.sql(f"SELECT MVT_ID_mvt FROM read_parquet('{template}')").df().MVT_ID_mvt
    missing, extra = set(ids) - set(pred.index), set(pred.index) - set(ids)
    if missing or extra:
        raise ValueError(f"id mismatch against the template: {len(missing)} missing, {len(extra)} extra")
    p = pred.reindex(ids.values)
    if p.isna().any():
        raise ValueError(f"{int(p.isna().sum())} null predictions")
    # The label itself goes to -12 s, so the clip is a judged option (C1: lane A tie H21, lane B win H22), not a physical law.
    vals = p.values if floor is None else np.clip(p.values, floor, None)
    sub = pd.DataFrame({"MVT_ID_mvt": ids.values.astype("float64"), "TAXITIME_SEC_mvt": np.rint(vals).astype("int32")})
    duckdb.from_df(sub).write_parquet(out)
    return sub


if __name__ == "__main__":
    # Re-emit a built file through the producer with a floor, so a floored candidate keeps its parent's fits (GPU refits do not reproduce).
    import hashlib
    import sys

    src, out, floor = sys.argv[1], sys.argv[2], float(sys.argv[3])
    parent = pd.read_parquet(src)
    pred = pd.Series(parent.TAXITIME_SEC_mvt.values.astype("float64"), index=parent.MVT_ID_mvt.values)
    sub = build_submission(pred, out, floor=floor)
    changed = int((sub.set_index("MVT_ID_mvt").TAXITIME_SEC_mvt != pred.reindex(sub.MVT_ID_mvt.values).values).sum())
    print(f"{out}: {len(sub)} rows, sha256 {hashlib.sha256(open(out, 'rb').read()).hexdigest()}, floor {floor:g} applied to {src} "
          f"(sha256 {hashlib.sha256(open(src, 'rb').read()).hexdigest()}): negative before {int((pred < 0).sum())}, after {int((sub.TAXITIME_SEC_mvt < 0).sum())}, rows changed {changed}")
