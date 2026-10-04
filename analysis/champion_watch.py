"""Keep a visible job record and continue the queued champion fit serially."""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from ctypes import wintypes
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from champion import write_record

RUN = ROOT / "data/runs/clean-champion-20261003"


def original_process(pid: int) -> tuple[object, object | None]:
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return kernel, None
    return kernel, handle


def process_running(kernel: object, handle: object | None) -> bool:
    if handle is None:
        return False
    result = wintypes.DWORD()
    if not kernel.GetExitCodeProcess(handle, ctypes.byref(result)):
        raise ctypes.WinError(ctypes.get_last_error())
    return result.value == 259


def snapshot(phase: str, pid: int | None, output: Path, status: str = "running", error: str | None = None,
             block: str = "1-7") -> None:
    complete = []
    pending = []
    for path in RUN.rglob("started.json"):
        component = path.parent
        if (component / "complete.json").exists():
            record = json.loads((component / "complete.json").read_text(encoding="utf-8"))
            complete.append({"component": str(component.relative_to(RUN)), "seconds": record["runtime_seconds"],
                             "checkpoint_replay": record["checkpoint_replay"]})
        elif component.relative_to(RUN).parts[0] == block:
            record = json.loads(path.read_text(encoding="utf-8"))
            pending.append({"component": str(component.relative_to(RUN)), "fit_rows": record["fit_rows"],
                            "fit_days": record["fit_days"], "stage": "full refit" if (component / "selection.json").exists() else "budget selection",
                            "started_file_mtime": path.stat().st_mtime})
    active = max(pending, key=lambda item: item["started_file_mtime"]) if pending else None
    record = {"status": status, "phase": phase, "pid": pid, "controller_pid": os.getpid(),
              "checked_utc": datetime.now(timezone.utc).isoformat(), "active_component": active,
              "completed_components": complete, "error": error,
              "block": block,
              "scope": f"Full original C/L/N donor and meta chain; block {block}. January/July labels remain excluded. No new scorer result yet."}
    write_record(RUN / "active.json", record)
    write_record(output / "prc26-active-run.json", record)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid", type=int)
    parser.add_argument("--months", choices=("1-7", "2-3", "9-10"), default="1-7")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.wait_pid is not None:
        kernel, handle = original_process(args.wait_pid)
        try:
            while process_running(kernel, handle):
                snapshot("Full CatBoost reference is training", args.wait_pid, args.output, block=args.months)
                time.sleep(15)
        finally:
            if handle is not None:
                kernel.CloseHandle(handle)
        first = RUN / args.months / "outer/ra_cb_wz_t7/complete.json"
        if not first.exists():
            snapshot("CatBoost fit needs repair", None, args.output, "failed", "First component has no completed replay-checked artifact.", args.months)
            raise RuntimeError("first CatBoost component did not complete")
    log = "pipeline" if args.months == "1-7" else f"pipeline-{args.months}"
    with (RUN / f"{log}.out").open("w", encoding="utf-8") as stdout, (RUN / f"{log}.err").open("w", encoding="utf-8") as stderr:
        child = subprocess.Popen([sys.executable, "-u", str(ROOT / "analysis/champion_screen.py"), "--months", args.months],
                                 cwd=ROOT, stdout=stdout, stderr=stderr, creationflags=subprocess.CREATE_NO_WINDOW)
        while child.poll() is None:
            snapshot(f"Full reference {args.months} is training", child.pid, args.output, block=args.months)
            time.sleep(15)
        code = child.returncode
    if code:
        snapshot("Nested fit needs repair", None, args.output, "failed", f"Pipeline exited with code {code}; saved components are preserved.", args.months)
        raise RuntimeError(f"champion pipeline failed: {code}")
    snapshot(f"Clean reference {args.months} is measured", None, args.output, "complete", block=args.months)


if __name__ == "__main__":
    main()
