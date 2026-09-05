import hashlib
import re

import duckdb
import numpy as np
import pandas as pd
import pytest

from taxi.data import DATA
from taxi.submit import TEMPLATE, build_submission


def template_ids() -> pd.Series:
    return duckdb.sql(f"SELECT MVT_ID_mvt FROM read_parquet('{TEMPLATE}')").df().MVT_ID_mvt


def test_data_hashes_match_protocol():
    with open("docs/EXPERIMENT-PROTOCOL.md", encoding="utf-8") as f:
        rows = re.findall(r"^([0-9a-f]{64})\s+(\S+\.parquet)$", f.read(), flags=re.M)
    assert len(rows) == 14
    for digest, name in rows:
        with open(f"{DATA}/{name}", "rb") as f:
            assert hashlib.sha256(f.read()).hexdigest() == digest, name


def test_submission_matches_template(tmp_path):
    ids = template_ids()
    out = (tmp_path / "merry-octopus_v0.parquet").as_posix()
    build_submission(pd.Series(900.4, index=ids.values), out)
    types = duckdb.sql(f"DESCRIBE SELECT * FROM read_parquet('{out}')").df()
    assert dict(zip(types.column_name, types.column_type)) == {"MVT_ID_mvt": "DOUBLE", "TAXITIME_SEC_mvt": "INTEGER"}
    back = duckdb.sql(f"SELECT * FROM read_parquet('{out}')").df()
    assert len(back) == len(ids) == 344841
    assert set(back.MVT_ID_mvt) == set(ids)
    assert back.TAXITIME_SEC_mvt.notna().all()
    assert (back.TAXITIME_SEC_mvt == 900).all()


def test_bad_predictions_raise(tmp_path):
    ids = template_ids()
    out = (tmp_path / "bad.parquet").as_posix()
    good = pd.Series(1.0, index=ids.values)
    with pytest.raises(ValueError, match="missing"):
        build_submission(good.iloc[1:], out)
    with pytest.raises(ValueError, match="extra"):
        build_submission(pd.concat([good, pd.Series([1.0], index=[-1.0])]), out)
    bad = good.copy()
    bad.iloc[0] = np.nan
    with pytest.raises(ValueError, match="null"):
        build_submission(bad, out)


def test_floor_is_off_by_default_and_clips_when_asked(tmp_path):
    ids = template_ids()
    out = (tmp_path / "clip.parquet").as_posix()
    p = pd.Series(300.0, index=ids.values)
    p.iloc[:5] = -40.0
    back = build_submission(p, out)
    assert (back.TAXITIME_SEC_mvt.iloc[:5] == -40).all() and (back.TAXITIME_SEC_mvt.iloc[5:] == 300).all()
    back = build_submission(p, out, floor=0)
    assert (back.TAXITIME_SEC_mvt.iloc[:5] == 0).all() and (back.TAXITIME_SEC_mvt.iloc[5:] == 300).all()
