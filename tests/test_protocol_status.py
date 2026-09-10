"""Section 4's status column is hand-written, and a lane that finishes updates its own page before its protocol row.

That drift runs the wrong way from the one Stage 2 warns about: a lane that RAN reads as open, so the campaign's own
not-done list nominates finished work. It cost a 1.5 h re-run of FAM1 on 2026-09-10. The same shape is already pinned
twice here, by `tests/test_model_class_register.py` and `tests/test_ceil_sources.py`; this is the third table.
"""
from __future__ import annotations

import io
import os
import re

import pytest

DOC = "docs/EXPERIMENT-PROTOCOL.md"
OPEN_MARKS = ("pending", "not yet run", "never run", "queued", "RUNNING")


def rows() -> list[tuple[int, str, str]]:
    """Every section-4 lane row as (line number, id, last non-empty cell). Cells are split on the pipe, and the id and
    the tail are read from the two ends because embedded pipes in the design column shift everything in between."""
    text = io.open(DOC, encoding="utf-8").read().splitlines()
    lo = next(i for i, ln in enumerate(text) if ln.startswith("## 4."))
    hi = next(i for i, ln in enumerate(text[lo + 1:], lo + 1) if ln.startswith("## "))
    out = []
    for i, ln in enumerate(text[lo:hi], lo + 1):
        if not ln.startswith("| ") or set(ln) <= set("|- "):
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        tail = next((c for c in reversed(cells[1:]) if c), "")
        out.append((i, cells[0].strip("*` "), tail))
    return out


def is_open(tail: str) -> bool:
    return any(m in tail for m in OPEN_MARKS)


OPEN = [r for r in rows() if is_open(r[2])]


def test_section_four_has_rows_at_all() -> None:
    """A parser that silently matches nothing turns this whole file into a pass."""
    assert len(rows()) > 40, f"{DOC} section 4 parsed to {len(rows())} rows, so the split is wrong"


@pytest.mark.parametrize("row", OPEN, ids=[f"{r[1]}:{r[0]}" for r in OPEN])
def test_a_lane_reported_open_has_produced_no_page(row: tuple[int, str, str]) -> None:
    """A lane's page is written by its own run, so the page existing is the run having happened."""
    line, lane, tail = row
    page = f"docs/{lane}.md"
    assert not os.path.exists(page), (
        f"{DOC}:{line} reports {lane} as open ({tail[:60]!r}) but {page} exists, so the lane ran and the row is stale")


RUN_MARKS = ("RUN ", "RAN ", "SCORED", "RUNNING")
RAN = [r for r in rows() if any(m in r[2] for m in RUN_MARKS) and not is_open(r[2])]


@pytest.mark.parametrize("row", RAN, ids=[f"{r[1]}:{r[0]}" for r in RAN])
def test_a_lane_reported_run_has_the_page_it_cites(row: tuple[int, str, str]) -> None:
    """The other direction: a result written before the run that produced it. A declaration NAMES the page it will
    write, so only a row already claiming a run is held to it."""
    line, lane, tail = row
    for page in sorted(set(re.findall(r"docs/[A-Za-z0-9_.-]+[.]md", tail))):
        assert os.path.exists(page), f"{DOC}:{line} reports {lane} as run and cites {page}, which does not exist"
