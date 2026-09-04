import duckdb
import pandas as pd

from taxi.data import DATA, features, month_file


def test_masked_training_equals_ranking_shaped_copy(tmp_path, feb_features):
    src, dst = month_file(2), (tmp_path / "shaped.parquet").as_posix()
    duckdb.sql(f"""COPY (SELECT * REPLACE (
        CASE WHEN PHASE_mvt = 'DEP' THEN NULL ELSE BLOCK_TIME_UTC_mvt END AS BLOCK_TIME_UTC_mvt,
        CASE WHEN PHASE_mvt = 'DEP' THEN NULL ELSE TAXITIME_SEC_mvt END AS TAXITIME_SEC_mvt)
        FROM read_parquet('{src}')) TO '{dst}' (FORMAT PARQUET)""")
    blanked = duckdb.sql(f"""SELECT count(BLOCK_TIME_UTC_mvt) + count(TAXITIME_SEC_mvt)
        FROM read_parquet('{dst}') WHERE PHASE_mvt = 'DEP'""").fetchone()[0]
    assert blanked == 0
    pd.testing.assert_frame_equal(features([dst]), feb_features)


def test_ranking_view_builds_every_departure(feb_features):
    f = features([f"{DATA}/ranking.parquet"])
    assert len(f) == 344841
    assert f.id.is_unique
    assert list(f.columns) == list(feb_features.columns)
