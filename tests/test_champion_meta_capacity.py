import time
from pathlib import Path

import pytest

import analysis.champion_meta_capacity as capacity
from analysis.champion_meta_capacity import deadline, native_check, source_check
from campaign import file_hash
from champion import write_record


def test_selected_native_rejects_changed_bytes(tmp_path: Path) -> None:
    directory = tmp_path / "nn3_pw_t7"
    directory.mkdir()
    path = directory / "selection.pt"
    path.write_bytes(b"selected weights")
    complete = {"artifact_hashes": {"selection.pt": file_hash(path)}}
    assert native_check(directory, {}, complete)["selection.pt"] == file_hash(path)
    path.write_bytes(b"different weights")
    with pytest.raises(ValueError, match="native bytes changed"):
        native_check(directory, {}, complete)


def test_recovery_cannot_authorize_another_component(tmp_path: Path) -> None:
    (tmp_path / "recovery.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="undeclared source recovery"):
        source_check(tmp_path, {"specification": {"engine_hashes": {}}}, {})


def test_declared_time_limits_stop_before_next_component() -> None:
    with pytest.raises(ValueError, match="30-minute"):
        deadline(time.monotonic() - 1801, 0)
    with pytest.raises(ValueError, match="15-minute"):
        deadline(time.monotonic(), 900)


def test_recovery_reads_exact_archive_and_checks_git_syntax(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    directory = tmp_path / "1-7/outer/ra_cb_wz_t7"
    directory.mkdir(parents=True)
    archive, current = directory / "original-train.py.txt", tmp_path / "train.py"
    archive.write_bytes(b"a = 1\r\nb = 2\n")
    current.write_bytes(b"a = 1\nb = 2\n")
    spec = {"engine_hashes": {"train.py": file_hash(archive)}, "versions": {}}
    resumed = {"train.py": file_hash(current)}
    complete = {"specification": {**spec, "engine_hashes": resumed}}
    recovery = {"original_specification": spec, "original_source_commit": "recorded-source",
                "resumed_engine_hashes": resumed, "completed_engine_compatibility": {
                    "execution_engine_hashes": resumed, "resumed_engine_hashes": resumed}}
    for name, record in (("recovery", recovery), ("selection", {"specification": spec}), ("complete", complete)):
        write_record(directory / f"{name}.json", record)
    monkeypatch.setattr(capacity, "ROOT", tmp_path)
    monkeypatch.setattr(capacity, "RUN", tmp_path)
    monkeypatch.setattr(capacity.subprocess, "check_output", lambda args: current.read_bytes())
    assert source_check(directory, {"specification": spec}, complete)["checks"] == "PASS"
    archive.write_bytes(b"a = 9\r\nb = 2\n")
    with pytest.raises(ValueError, match="original recovery source changed"):
        source_check(directory, {"specification": spec}, complete)
