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
    count = 0
    for path in root.rglob("complete.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        name = record.get("specification", {}).get("name", "")
        if name.startswith(("ra_cb", "nn", "mix_cb")):
            used += record["runtime_seconds"]
            count += 1
    record = {"limit_seconds": 44 * 3600, "used_seconds": used, "completed_gpu_components": count,
              "accounting": "Saved GPU-family component wall time, including selection/refit/replay; conservative GPU-time proxy.",
              "status": "within budget" if used < 44 * 3600 else "budget reached"}
    write_record(root / "budget.json", record)
    if used >= record["limit_seconds"]:
        raise RuntimeError("clean champion GPU allowance reached at a saved-component boundary")
    return record
