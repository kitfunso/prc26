"""scripts/queue-jobs.json is the second hand-maintained list of lanes, and nothing tied it to the first.

O1 sat in it with no done marker for three days after section 4 marked it SUPERSEDED by O2, so any plain
`python scripts/run_queue.py` would have started it and taken the card for hours. The protocol row is the source; this pins
the queue to it. The marker is a CELL that opens with the word, not the phrase "never run", which four rows use as prose.
"""
from __future__ import annotations

import io
import json

import pytest

DOC = "docs/EXPERIMENT-PROTOCOL.md"
JOBS = "scripts/queue-jobs.json"
MARK = "SUPERSEDED"


def banned() -> set[str]:
    """Lane ids whose own section-4 row carries a cell opening with SUPERSEDED."""
    out = set()
    for ln in io.open(DOC, encoding="utf-8").read().splitlines():
        if not ln.startswith("| ") or set(ln) <= set("|- "):
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if any(c.startswith(MARK) for c in cells[1:]):
            out.add(cells[0].strip("*` ").lower())
    return out


NAMES = [n for n, _ in json.load(io.open(JOBS, encoding="utf-8"))]


def test_the_queue_parses_to_jobs_at_all() -> None:
    """A parser that matches nothing turns the check below into a pass."""
    assert len(NAMES) > 10, f"{JOBS} parsed to {len(NAMES)} jobs, so the shape changed"
    assert len(set(NAMES)) == len(NAMES), f"{JOBS} names a job twice: {sorted(NAMES)}"


@pytest.mark.parametrize("name", NAMES)
def test_no_queued_job_is_a_lane_the_protocol_forbids(name: str) -> None:
    assert name not in banned(), f"{JOBS} queues {name}, which {DOC} records as never to run"
