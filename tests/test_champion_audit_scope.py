from pathlib import Path

import pandas as pd
import pytest

from analysis import champion_checkpoint_audit as checkpoint
from analysis import champion_donor_audit as donor
from analysis import champion_evaluation_audit as evaluation
from campaign import file_hash, outer_masks
from champion import configuration, frame_hash, ids_hash, write_record
from champion_neighbours import file_month, month_groups
from train import stopping_split


@pytest.fixture
def cold_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> pd.DataFrame:
    rows = []
    for month in range(1, 13):
        start = pd.Timestamp(2025, month, 1)
        end = start + pd.offsets.MonthBegin(1)
        for day in range(10):
            for lane in (0, 1):
                rows.append({"id": len(rows), "file": f"training_{start:%Y-%m-%d}_{end:%Y-%m-%d}.parquet",
                             "day": start + pd.Timedelta(days=day), "apt": "AAA", "runway": "01",
                             "has_plan": lane, "z_aobt": 100., "t_off": 200., "y": 120.})
    frame = pd.DataFrame(rows)
    source = tmp_path / "data/runs/contender-20261003"
    source.mkdir(parents=True)
    path = source / "cold_features.parquet"
    frame.to_parquet(path)
    write_record(source / "inputs.json", {"feature_hash": file_hash(path)})
    run = tmp_path / "data/runs/clean-champion-20261003"
    monkeypatch.setattr(donor, "ROOT", tmp_path)
    monkeypatch.setattr(donor, "RUN", run)
    monkeypatch.setattr(checkpoint, "RUN", run)
    monkeypatch.setattr(checkpoint, "ROOT", tmp_path)
    return frame


@pytest.mark.parametrize("months", [(2, 3), (9, 10)])
def test_donor_fit_excludes_complete_outer_files(cold_rows: pd.DataFrame, months: tuple[int, ...]) -> None:
    frame, fit, _, _ = donor.input_rows(False, (), months)
    expected = set(range(1, 13)) - {1, 7, *months}
    assert set(file_month(frame[fit])) == expected
    assert frame[fit].has_plan.eq(1).all()
    assert set(frame.loc[fit, "id"]) == set(cold_rows.loc[outer_masks(cold_rows, months)[0] & cold_rows.has_plan.eq(1), "id"])


def test_inner_group_is_derived_from_the_active_outer_block(cold_rows: pd.DataFrame) -> None:
    frame, fit, _, _ = donor.input_rows(False, (4, 8, 11), (2, 3))
    assert set(file_month(frame[fit])) == {5, 6, 9, 10, 12}
    with pytest.raises(ValueError, match="not a declared complete-file group"):
        donor.input_rows(False, (2, 5, 9, 12), (2, 3))


def test_stopping_days_stay_out_of_the_active_donor(cold_rows: pd.DataFrame) -> None:
    frame, full, _, _ = donor.input_rows(False, (4, 8, 11), (2, 3))
    inner, stop = stopping_split(frame[full])
    _, fit, _, excluded = donor.input_rows(True, (4, 8, 11), (2, 3))
    assert set(frame.loc[fit, "id"]) == set(inner.id)
    assert not frame[fit].day.isin(stop.day).any()
    assert excluded["rows"] == len(stop)


def test_leaked_confirmation_file_rejected_before_donor_read(cold_rows: pd.DataFrame, tmp_path: Path) -> None:
    fit = cold_rows.has_plan.eq(1)
    with pytest.raises(ValueError, match="held-out IDs or files"):
        donor.scope_values(cold_rows, fit, {"group": None}, tmp_path, "unused", (2, 3))


def test_confirmation_donor_cannot_use_a_first_block_proof(cold_rows: pd.DataFrame) -> None:
    proof = donor.proof_directory((2, 3)) / "first-donor-audit.json"
    write_record(proof, {"months": [1, 7]})
    with pytest.raises(ValueError, match="another outer block"):
        checkpoint.donor_windows(cold_rows, cold_rows, False, (2, 3))


def test_confirmation_meta_cannot_use_a_first_block_proof(cold_rows: pd.DataFrame) -> None:
    train = cold_rows[outer_masks(cold_rows, (2, 3))[0]]
    groups = month_groups(train, pd.Index(train.id))
    tag = "meta-" + "-".join(map(str, groups[0]))
    directory = donor.proof_directory((2, 3))
    write_record(directory / f"{tag}-assembly-audit.json", {"months": [1, 7], "meta_group": groups[0]})
    complete = {"months": [2, 3], "meta_groups": [list(group) for group in groups]}
    with pytest.raises(ValueError, match="another outer block or group"):
        evaluation.checked_meta(train, pd.DataFrame(), directory, complete)


def test_proof_paths_preserve_the_checked_first_block(cold_rows: pd.DataFrame) -> None:
    assert donor.proof_directory((1, 7)) == donor.RUN
    assert donor.proof_directory((2, 3)) == donor.RUN / "2-3"
    with pytest.raises(ValueError, match="not declared"):
        donor.proof_directory((4, 5))


def test_catboost_fingerprints_use_confirmation_rows_without_donors(cold_rows: pd.DataFrame) -> None:
    frame = cold_rows.assign(apt=cold_rows.apt.astype("category"))
    fit, query = outer_masks(frame, (2, 3))
    fit, query = fit & frame.has_plan.eq(1), query & frame.has_plan.eq(1)
    columns = ["id", "z_aobt", "apt"]
    spec = {"config": {"cat": ["apt"], "num": ["z_aobt"]},
            "fit_input_hash": frame_hash(frame[fit], [*columns, "y"]),
            "query_input_hash": frame_hash(frame[query], columns),
            "selection_input_hash": frame_hash(frame[fit], [*columns, "y"])}
    proof = checkpoint.input_fingerprints({"specification": spec}, (2, 3))
    assert proof["checks"] == "PASS" and proof["full_donor"] == proof["selection_donor"] == {}
    spec["fit_input_hash"] = frame_hash(frame[query], [*columns, "y"])
    with pytest.raises(ValueError, match="feature fingerprints changed"):
        checkpoint.input_fingerprints({"specification": spec}, (2, 3))


@pytest.mark.parametrize("changed", ["source", "model"])
def test_donor_rejects_changed_native_bytes(cold_rows: pd.DataFrame, changed: str) -> None:
    fit, query = outer_masks(cold_rows, (2, 3))
    train, test = cold_rows[fit & cold_rows.has_plan.eq(1)], cold_rows[query & cold_rows.has_plan.eq(1)]
    inner, stop = stopping_split(train)
    path = donor.RUN / "2-3/donors/full/ra_lgb_wz"
    path.mkdir(parents=True)
    model = path / "body.model"
    model.write_bytes(b"saved model bytes")
    source = donor.ROOT / "engine.py"
    source.write_bytes(b"saved fitting source")
    record = {"checkpoint_replay": "PASS", "fit_rows": len(train), "fit_days": train.day.nunique(),
              "inner_fit_rows": len(inner), "stopping_rows": len(stop),
              "inner_fit_ids_hash": ids_hash(inner.id), "stopping_ids_hash": ids_hash(stop.id),
              "specification": {"name": path.name, "config": configuration(path.name),
                                "fit_ids": ids_hash(train.id), "query_ids": ids_hash(test.id),
                                "provenance": {"cold_feature_hash": "cold bytes"},
                                "engine_hashes": {"engine.py": file_hash(source)}},
              "artifact_hashes": {model.name: file_hash(model)}}
    write_record(path / "complete.json", record)
    (source if changed == "source" else model).write_bytes(b"changed bytes")
    with pytest.raises(ValueError, match="fitting source bytes changed|native artifact bytes changed"):
        donor.member_values(path, {}, train, test, "cold bytes", True)
