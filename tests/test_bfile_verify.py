"""A ranking file carries lane B from its parent untouched. `bfile.verify` is what makes that a check rather than a claim."""
import numpy as np
import pandas as pd
import pytest

from taxi.bfile import verify

IDS = np.arange(6)
LANE_A = np.array([True, True, True, True, False, False])


def _write(tmp_path, name, vals):
    p = tmp_path / name
    pd.DataFrame({"MVT_ID_mvt": IDS, "TAXITIME_SEC_mvt": np.asarray(vals, dtype="float64")}).to_parquet(p, index=False)
    return str(p)


def test_a_moved_lane_b_row_is_caught(tmp_path):
    parent = _write(tmp_path, "parent.parquet", [100, 200, 300, 400, 500, 600])
    out = _write(tmp_path, "out.parquet", [110, 210, 310, 410, 500, 601])  # the last row is lane B and it moved
    with pytest.raises(ValueError, match="lane-B rows moved"):
        verify(out, parent, IDS, LANE_A)


def test_a_clean_child_passes_and_says_what_moved(tmp_path):
    parent = _write(tmp_path, "parent.parquet", [100, 200, 300, 400, 500, 600])
    out = _write(tmp_path, "out.parquet", [110, 210, 310, 400, 500, 600])
    assert "lane A moved on 3 of 4" in verify(out, parent, IDS, LANE_A)


@pytest.mark.parametrize("vals, why", [
    ([110, 210, 310, 410, 500, np.nan], "NaN or infinite"),
    ([110, 210, 310, -1, 500, 600], "below the declared floor"),
])
def test_an_unusable_value_is_caught(tmp_path, vals, why):
    parent = _write(tmp_path, "parent.parquet", [100, 200, 300, 400, 500, 600])
    with pytest.raises(ValueError, match=why):
        verify(_write(tmp_path, "out.parquet", vals), parent, IDS, LANE_A)
