from pathlib import Path

import pytest

from champion import write_record
from champion_budget import budget_guard


def component(root: Path, block: str, name: str, seconds: float) -> None:
    write_record(root / block / name / "complete.json",
                 {"specification": {"name": name}, "runtime_seconds": seconds})


def test_block_limit_excludes_cpu_and_other_blocks(tmp_path: Path) -> None:
    root = tmp_path / "clean-champion-20261003"
    component(root, "1-7", "ra_cb_wz", 40000.)
    component(root, "2-3", "nn3_pw_t7", 3600.)
    component(root, "2-3", "ra_lgb_pw_t7", 90000.)
    write_record(root / "2-3/budget-limit.json", {"gpu_family_limit_seconds": 7200})
    record = budget_guard(root / "2-3/outer")
    assert record["used_seconds"] == 43600.
    assert record["block"]["used_seconds"] == 3600.
    assert record["status"] == "within budget"


def test_block_limit_stops_at_saved_boundary(tmp_path: Path) -> None:
    root = tmp_path / "clean-champion-20261003"
    component(root, "2-3", "nn3_pw_t7", 7200.)
    write_record(root / "2-3/budget-limit.json", {"gpu_family_limit_seconds": 7200})
    with pytest.raises(RuntimeError, match="block 2-3 GPU allocation reached"):
        budget_guard(root / "2-3/donors/full")


def test_global_limit_still_stops_with_a_block_allocation(tmp_path: Path) -> None:
    root = tmp_path / "clean-champion-20261003"
    component(root, "1-7", "ra_cb_wz", 140000.)
    component(root, "2-3", "nn3_pw_t7", 20000.)
    write_record(root / "2-3/budget-limit.json", {"gpu_family_limit_seconds": 43200})
    with pytest.raises(RuntimeError, match="clean champion GPU allowance reached"):
        budget_guard(root / "2-3/outer")


def test_invalid_block_allocation_rejected(tmp_path: Path) -> None:
    root = tmp_path / "clean-champion-20261003"
    write_record(root / "2-3/budget-limit.json", {"gpu_family_limit_seconds": -1})
    with pytest.raises(ValueError, match="block GPU allocation must be positive"):
        budget_guard(root / "2-3/outer")
