"""Check the remaining campaign GPU envelope between saved components."""
from __future__ import annotations

import json
from pathlib import Path

from champion import write_record


def budget_guard(directory: Path) -> dict:
    root = next((path for path in (directory, *directory.parents) if path.name == "clean-champion-20261003"), None)
    if root is None:
        return {}
    used = 0.
    block_used = 0.
    count = 0
    parts = directory.relative_to(root).parts
    block = parts[0] if parts else None
    allocation = root / block / "budget-limit.json" if block else None
    block_limit = None
    if allocation is not None and allocation.exists():
        block_limit = float(json.loads(allocation.read_text(encoding="utf-8"))["gpu_family_limit_seconds"])
        if not 0 < block_limit <= 44 * 3600:
            raise ValueError("block GPU allocation must be positive and within the campaign allowance")
    for path in root.rglob("complete.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        name = record.get("specification", {}).get("name", "")
        if name.startswith(("ra_cb", "nn", "mix_cb")):
            used += record["runtime_seconds"]
            count += 1
            if path.relative_to(root).parts[0] == block:
                block_used += record["runtime_seconds"]
    record = {"limit_seconds": 44 * 3600, "used_seconds": used, "completed_gpu_components": count,
              "accounting": "Saved GPU-family component wall time, including selection/refit/replay; conservative GPU-time proxy.",
              "status": "within budget" if used < 44 * 3600 else "budget reached"}
    if block_limit is not None:
        record["block"] = {"months": block, "used_seconds": block_used, "limit_seconds": block_limit}
        if block_used >= block_limit:
            record["status"] = "block budget reached"
    write_record(root / "budget.json", record)
    if used >= record["limit_seconds"]:
        raise RuntimeError("clean champion GPU allowance reached at a saved-component boundary")
    if block_limit is not None and block_used >= block_limit:
        raise RuntimeError(f"block {block} GPU allocation reached at a saved-component boundary")
    return record
