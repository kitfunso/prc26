"""Submission checks with mocked parquet I/O and S3; no challenge data or keys needed.

Run with python -m unittest discover -s tests -p test_board.py (also under -O).
"""
import contextlib
import hashlib
import io
import os
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

import board


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.template = pd.DataFrame({"MVT_ID_mvt": [11, 22], "TAXITIME_SEC_mvt": [np.nan, np.nan]})
        self.submission = pd.DataFrame({"MVT_ID_mvt": [11, 22], "TAXITIME_SEC_mvt": [0., 1200.]})

    def check(self, sub, template=None):
        template = self.template if template is None else template
        with patch.object(board.pd, "read_parquet", side_effect=[sub, template]), \
                patch.object(board.Path, "read_bytes", return_value=b"mock parquet"), \
                contextlib.redirect_stdout(io.StringIO()):
            return board.check("merry-octopus_v63.parquet")

    def test_valid_predictions(self):
        self.assertIs(self.check(self.submission), self.submission)

    def test_id_order_need_not_match_template(self):
        reordered = self.submission.iloc[::-1]
        self.assertIs(self.check(reordered), reordered)

    def test_missing_or_extra_ids(self):
        for ids in ([11], [11, 33], [11, 22, 33]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                self.check(pd.DataFrame({"MVT_ID_mvt": ids, "TAXITIME_SEC_mvt": [1.] * len(ids)}))

    def test_duplicate_or_null_ids(self):
        for ids in ([11, 11], [11, None], [11, pd.NA]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                self.check(pd.DataFrame({"MVT_ID_mvt": ids, "TAXITIME_SEC_mvt": [1., 2.]}))

    def test_prediction_values(self):
        for value in (np.nan, np.inf, -np.inf, -1.):
            sub = self.submission.copy()
            sub.loc[0, "TAXITIME_SEC_mvt"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.check(sub)

    def test_nullable_numeric_predictions(self):
        sub = self.submission.copy()
        sub["TAXITIME_SEC_mvt"] = pd.Series([pd.NA, 1200.], dtype="Float64")
        with self.assertRaises(ValueError):
            self.check(sub)
        sub["TAXITIME_SEC_mvt"] = pd.Series([0., 1200.], dtype="Float64")
        self.assertIs(self.check(sub), sub)

    def test_prediction_dtype(self):
        for values in (["0", "1200"], [False, True], [0j, 1200 + 1j]):
            sub = self.submission.copy()
            sub["TAXITIME_SEC_mvt"] = values
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.check(sub)

    def test_columns(self):
        for sub in (self.submission.iloc[:, ::-1], self.submission.drop(columns="MVT_ID_mvt"),
                    self.submission.assign(extra=1)):
            with self.subTest(columns=list(sub)), self.assertRaises(ValueError):
                self.check(sub)

    def test_invalid_templates(self):
        templates = [self.template.iloc[:0], self.template.assign(MVT_ID_mvt=[11, 11]),
                     self.template.assign(MVT_ID_mvt=[11, None]), self.template.assign(extra=1)]
        for template in templates:
            with self.subTest(template=template), self.assertRaises(ValueError):
                self.check(self.submission, template)

    def test_invalid_name_blocks_s3(self):
        with patch.object(board, "check") as check, patch.object(board, "_s3") as s3:
            with self.assertRaises(ValueError):
                board.upload("wrong-name.parquet")
            check.assert_not_called()
            s3.assert_not_called()

    def test_invalid_file_blocks_s3(self):
        with patch.object(board, "check", side_effect=ValueError("invalid")), \
                patch.object(board, "_s3") as s3:
            with self.assertRaises(ValueError):
                board.upload("merry-octopus_v63.parquet")
            s3.assert_not_called()

    def test_valid_upload_and_readback(self):
        payload = b"mock parquet"
        s3 = MagicMock()
        s3.head_object.return_value = {"ContentLength": len(payload),
                                       "ETag": '"' + hashlib.md5(payload).hexdigest() + '"'}
        with patch.object(board, "check"), patch.object(board, "_s3", return_value=s3), \
                patch.object(board.Path, "read_bytes", return_value=payload), \
                contextlib.redirect_stdout(io.StringIO()):
            board.upload("merry-octopus_v63.parquet")
        s3.put_object.assert_called_once_with(Bucket=board.BUCKET, Key="merry-octopus_v63.parquet", Body=payload)
        s3.head_object.assert_called_once_with(Bucket=board.BUCKET, Key="merry-octopus_v63.parquet")

    def test_bad_upload_readback(self):
        for head in ({"ContentLength": 0, "ETag": '"bad"'},
                     {"ContentLength": 12, "ETag": '"bad"'}):
            s3 = MagicMock()
            s3.head_object.return_value = head
            with self.subTest(head=head), patch.object(board, "check"), \
                    patch.object(board, "_s3", return_value=s3), \
                    patch.object(board.Path, "read_bytes", return_value=b"mock parquet"):
                with self.assertRaises(RuntimeError):
                    board.upload("merry-octopus_v63.parquet")


class CredentialTests(unittest.TestCase):
    def test_missing_or_empty_credentials(self):
        for env in ({}, {"PRC_KEY": "x"}, {"PRC_KEY": "x", "PRC_SECRET": " "}):
            with self.subTest(env=env), patch.dict(os.environ, env, clear=True), \
                    patch.object(board.Path, "exists", return_value=False):
                with self.assertRaisesRegex(ValueError, "PRC_KEY and PRC_SECRET"):
                    board._s3()

    def connect(self, env, file_content=None):
        boto3, config = MagicMock(), MagicMock()
        with patch.dict(os.environ, env, clear=True), \
                patch.object(board.Path, "exists", return_value=file_content is not None), \
                patch.object(board.Path, "read_text", return_value=file_content or ""), \
                patch.dict("sys.modules", {"boto3": boto3, "botocore": MagicMock(), "botocore.config": config}):
            board._s3()
        return boto3.client.call_args.kwargs

    def test_environment_only(self):
        args = self.connect({"PRC_KEY": "env-key", "PRC_SECRET": "env-secret"})
        self.assertEqual(args["aws_access_key_id"], "env-key")
        self.assertEqual(args["aws_secret_access_key"], "env-secret")

    def test_env_file_and_environment_precedence(self):
        args = self.connect({"PRC_KEY": "env-key"},
                            "# keys\n\ninvalid-line\nPRC_KEY=file-key\nPRC_SECRET=file-secret\n")
        self.assertEqual(args["aws_access_key_id"], "env-key")
        self.assertEqual(args["aws_secret_access_key"], "file-secret")


if __name__ == "__main__":
    unittest.main()
