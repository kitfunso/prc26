"""Sequential detached runner for the not-done queue (docs/NOTDONE-PLAN.md): one GPU job at a time, each logged under data/queue/, a failure is
recorded and the queue moves on. A finished job leaves data/queue/<name>.done, so a relaunch resumes."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

QUEUE = "data/queue"
JOBS = [("oof", ["-m", "taxi.champ", "oof", "MC2", "D1"]), ("e2", ["-m", "taxi.e2"]), ("sb1", ["-m", "taxi.sb1"]),
        ("ws1", ["-m", "taxi.lanes", "WS1"]), ("t1", ["-m", "taxi.lanes", "T1"]), ("g1", ["-m", "taxi.lanes", "G1"]),
        ("tw1", ["-m", "taxi.lanes", "TW1"]), ("r1", ["-m", "taxi.lanes", "R1"]), ("h1", ["-m", "taxi.lanes", "H1"]),
        ("ff", ["-m", "taxi.fwd"]), ("m3", ["-m", "taxi.m3"]), ("e1", ["-m", "taxi.e1"]), ("o1", ["-m", "taxi.o1"]),
        ("th1", ["-m", "taxi.lanes", "TH1"]),
    ("q1", ["-m", "taxi.lanes", "Q1"])]


def main() -> None:
    os.makedirs(QUEUE, exist_ok=True)
    only = sys.argv[1:]
    for name, args in JOBS:
        if (only and name not in only) or os.path.exists(f"{QUEUE}/{name}.done"):
            continue
        t0 = time.time()
        with open(f"{QUEUE}/{name}.log", "a", encoding="utf-8") as out:
            rc = subprocess.call([sys.executable, "-u", *args], stdout=out, stderr=subprocess.STDOUT, cwd=os.getcwd())
        status = dict(job=name, args=args, returncode=rc, minutes=round((time.time() - t0) / 60, 1), finished=time.strftime("%Y-%m-%d %H:%M:%S"))
        with open(f"{QUEUE}/{name}.{'done' if rc == 0 else 'failed'}", "w", encoding="utf-8") as fh:
            json.dump(status, fh)
        print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
