"""Sequential detached runner for the not-done queue (docs/NOTDONE-PLAN.md): one GPU job at a time, each logged under data/queue/, a failure is
recorded and the queue moves on. A finished job leaves data/queue/<name>.done, so a relaunch resumes."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

QUEUE = "data/queue"
JOBS_FILE = "scripts/queue-jobs.json"


def pending(only: list[str]) -> tuple[str, list[str]] | None:
    """Re-read every iteration, so a job appended while the queue runs is picked up without a relaunch; a failure leaves a marker and is skipped."""
    with open(JOBS_FILE, encoding="utf-8") as fh:
        for name, args in json.load(fh):
            if only and name not in only:
                continue
            if any(os.path.exists(f"{QUEUE}/{name}.{s}") for s in ("done", "failed")):
                continue
            return name, args
    return None


def main() -> None:
    os.makedirs(QUEUE, exist_ok=True)
    only = sys.argv[1:]
    while (job := pending(only)) is not None:
        name, args = job
        t0 = time.time()
        with open(f"{QUEUE}/{name}.log", "a", encoding="utf-8") as out:
            rc = subprocess.call([sys.executable, "-u", *args], stdout=out, stderr=subprocess.STDOUT, cwd=os.getcwd())
        status = dict(job=name, args=args, returncode=rc, minutes=round((time.time() - t0) / 60, 1), finished=time.strftime("%Y-%m-%d %H:%M:%S"))
        with open(f"{QUEUE}/{name}.{'done' if rc == 0 else 'failed'}", "w", encoding="utf-8") as fh:
            json.dump(status, fh)
        print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
