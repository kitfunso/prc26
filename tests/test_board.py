"""Submission failures must stop before S3 access, including under python -O."""
from __future__ import annotations

import hashlib
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import board


@pytest.fixture
def submission() -> pd.DataFrame:
    return pd.DataFrame({"MVT_ID_mvt": [11, 22], "TAXITIME_SEC_mvt": [0., 1200.]})


def check(sub: pd.DataFrame, template: pd.DataFrame | None = None) -> pd.DataFrame:
    if template is None:
        template = pd.DataFrame({"MVT_ID_mvt": [11, 22], "TAXITIME_SEC_mvt": [np.nan, np.nan]})
    with patch.object(board.pd, "read_parquet", side_effect=[sub, template]), \
            patch.object(board.Path, "read_bytes", return_value=b"mock parquet"):
        return board.check("merry-octopus_v63.parquet")


def test_valid_predictions_and_reordered_ids(submission: pd.DataFrame) -> None:
    assert check(submission) is submission
    reordered = submission.iloc[::-1]
    assert check(reordered) is reordered


@pytest.mark.parametrize("ids", ([11], [11, 33], [11, 22, 33], [11, 11], [11, None], [11, pd.NA]))
def test_invalid_row_sets(ids: list) -> None:
    with pytest.raises(ValueError):
        check(pd.DataFrame({"MVT_ID_mvt": ids, "TAXITIME_SEC_mvt": [1.] * len(ids)}))


@pytest.mark.parametrize("value", (np.nan, np.inf, -np.inf, -1.))
def test_invalid_prediction_values(submission: pd.DataFrame, value: float) -> None:
    submission.loc[0, "TAXITIME_SEC_mvt"] = value
    with pytest.raises(ValueError):
        check(submission)


@pytest.mark.parametrize("values", (["0", "1200"], [False, True], [0j, 1200 + 1j]))
def test_invalid_prediction_types(submission: pd.DataFrame, values: list) -> None:
    submission["TAXITIME_SEC_mvt"] = values
    with pytest.raises(ValueError):
        check(submission)


def test_nullable_numeric_predictions(submission: pd.DataFrame) -> None:
    submission["TAXITIME_SEC_mvt"] = pd.Series([pd.NA, 1200.], dtype="Float64")
    with pytest.raises(ValueError):
        check(submission)
    submission["TAXITIME_SEC_mvt"] = pd.Series([0., 1200.], dtype="Float64")
    assert check(submission) is submission


def test_invalid_columns_and_templates(submission: pd.DataFrame) -> None:
    for sub in (submission.iloc[:, ::-1], submission.drop(columns="MVT_ID_mvt"), submission.assign(extra=1)):
        with pytest.raises(ValueError):
            check(sub)
    for template in (submission.iloc[:0], submission.assign(MVT_ID_mvt=[11, 11]),
                     submission.assign(MVT_ID_mvt=[11, None]), submission.assign(extra=1)):
        with pytest.raises(ValueError):
            check(submission, template)


def test_invalid_name_or_file_blocks_s3() -> None:
    with patch.object(board, "check") as validate, patch.object(board, "_s3") as s3:
        with pytest.raises(ValueError):
            board.upload("wrong-name.parquet")
        validate.assert_not_called()
        s3.assert_not_called()
        validate.side_effect = ValueError("invalid")
        with pytest.raises(ValueError):
            board.upload("merry-octopus_v63.parquet")
        s3.assert_not_called()


def test_valid_upload_and_readback() -> None:
    payload, s3 = b"mock parquet", MagicMock()
    s3.head_object.return_value = {"ContentLength": len(payload), "ETag": '"' + hashlib.md5(payload).hexdigest() + '"'}
    with patch.object(board, "check"), patch.object(board, "_s3", return_value=s3), \
            patch.object(board.Path, "read_bytes", return_value=payload):
        board.upload("merry-octopus_v63.parquet")
    s3.put_object.assert_called_once_with(Bucket=board.BUCKET, Key="merry-octopus_v63.parquet", Body=payload)
    s3.head_object.assert_called_once_with(Bucket=board.BUCKET, Key="merry-octopus_v63.parquet")


@pytest.mark.parametrize("head", ({"ContentLength": 0, "ETag": '"bad"'}, {"ContentLength": 12, "ETag": '"bad"'}))
def test_bad_upload_readback(head: dict) -> None:
    s3 = MagicMock()
    s3.head_object.return_value = head
    with patch.object(board, "check"), patch.object(board, "_s3", return_value=s3), \
            patch.object(board.Path, "read_bytes", return_value=b"mock parquet"), pytest.raises(RuntimeError):
        board.upload("merry-octopus_v63.parquet")


@pytest.mark.parametrize("env", ({}, {"PRC_KEY": "test"}, {"PRC_KEY": "test", "PRC_SECRET": " "}))
def test_missing_credentials(env: dict) -> None:
    with patch.dict(board.os.environ, env, clear=True), patch.object(board.Path, "exists", return_value=False), \
            pytest.raises(ValueError, match="PRC_KEY and PRC_SECRET"):
        board._s3()


@pytest.mark.parametrize("file_content", (None, "# keys\n\ninvalid-line\nPRC_KEY=file-key\nPRC_SECRET=file-secret\n"))
def test_environment_credentials_take_precedence(file_content: str | None) -> None:
    boto3, config = MagicMock(), MagicMock()
    with patch.dict(board.os.environ, {"PRC_KEY": "env-key", "PRC_SECRET": "env-secret"}, clear=True), \
            patch.object(board.Path, "exists", return_value=file_content is not None), \
            patch.object(board.Path, "read_text", return_value=file_content or ""), \
            patch.dict("sys.modules", {"boto3": boto3, "botocore": MagicMock(), "botocore.config": config}):
        board._s3()
    assert boto3.client.call_args.kwargs["aws_access_key_id"] == "env-key"
    assert boto3.client.call_args.kwargs["aws_secret_access_key"] == "env-secret"


def test_env_file_supplies_missing_environment_credential() -> None:
    boto3, config = MagicMock(), MagicMock()
    with patch.dict(board.os.environ, {"PRC_KEY": "env-key"}, clear=True), \
            patch.object(board.Path, "exists", return_value=True), \
            patch.object(board.Path, "read_text", return_value="PRC_KEY=file-key\nPRC_SECRET=file-secret\n"), \
            patch.dict("sys.modules", {"boto3": boto3, "botocore": MagicMock(), "botocore.config": config}):
        board._s3()
    assert boto3.client.call_args.kwargs["aws_access_key_id"] == "env-key"
    assert boto3.client.call_args.kwargs["aws_secret_access_key"] == "file-secret"
