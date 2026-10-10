"""Remote failure evidence must remain visible without relaxing successful-score checks."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from analysis import release_readiness_20261008 as release


@pytest.fixture
def final_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[dict, MagicMock]:
    preparation = tmp_path / 'preparation'
    preparation.mkdir()
    receipt = {'final_path': 'final.parquet', 'rows': 2, 'bytes': 5,
               'sha256': hashlib.sha256(b'final').hexdigest(),
               'prior_scored_object': {'scorer_result': {'score': 270.43}}}
    (preparation / 'final-designation-20261008.json').write_text(json.dumps(receipt))
    client = MagicMock()
    monkeypatch.setattr(release, 'RUN', tmp_path)
    monkeypatch.setattr(release.board, 'check', MagicMock())
    monkeypatch.setattr(release.board, '_s3', lambda: client)
    return receipt, client


def result(status: str = 'Succeeded') -> dict:
    value = {'status': status, 'file': 'merry-octopus_final.parquet', 'bucket': release.board.BUCKET}
    if status == 'Succeeded':
        value.update(used_pairs=2, score=270.43)
    else:
        value.update(error='s3://prc-2026-datasets/dev_sub_template.parquet', error_type='FileNotFoundError')
    return value


def responses(client: MagicMock, value: dict, payload: bytes = b'final') -> None:
    client.get_object.side_effect = [{'Body': io.BytesIO(payload)},
                                     {'Body': io.BytesIO(json.dumps(value).encode())}]


def test_failed_result_is_preserved_without_score_acceptance(final_case: tuple[dict, MagicMock]) -> None:
    _, client = final_case
    failed = result('Failed')
    responses(client, failed)
    record = release.final_verification()
    assert record['result'] == failed
    assert record['scorer_status'] == 'Failed'
    assert record['score_verified'] is False
    client.put_object.assert_not_called()


def test_identical_success_remains_checked(final_case: tuple[dict, MagicMock]) -> None:
    _, client = final_case
    responses(client, result())
    assert release.final_verification()['score_verified'] is True


@pytest.mark.parametrize('change', ({'used_pairs': 3}, {'score': 270.42}, {'file': 'other.parquet'},
                                    {'bucket': 'other-bucket'}, {'status': 'Unknown'}))
def test_mismatched_success_still_fails(final_case: tuple[dict, MagicMock], change: dict) -> None:
    _, client = final_case
    value = result() | change
    responses(client, value)
    with pytest.raises(ValueError):
        release.final_verification()


def test_failure_without_error_evidence_is_rejected(final_case: tuple[dict, MagicMock]) -> None:
    _, client = final_case
    value = result('Failed')
    del value['error']
    responses(client, value)
    with pytest.raises(ValueError, match='error evidence'):
        release.final_verification()


def test_changed_stored_bytes_still_fail(final_case: tuple[dict, MagicMock]) -> None:
    _, client = final_case
    responses(client, result('Failed'), b'other')
    with pytest.raises(ValueError, match='stored final artifact changed'):
        release.final_verification()
