"""The promotion floor is denominated in BOARD seconds, so the two lanes buy the same thing with a holdout gain."""
from __future__ import annotations

import io
import os
import re

import pytest

from taxi.ceil import PER_A, PER_B
from taxi.verdict import BOARD_FLOOR, Verdict, promote_floor


def test_lane_a_bar_is_unchanged_at_two_holdout_seconds():
    """BOARD_FLOOR is pinned to what lane A already asked for, so no lane-A verdict in the ledger moves."""
    assert promote_floor("A") == pytest.approx(2.0)
    assert BOARD_FLOOR == pytest.approx(2.0 * PER_A)


def test_both_lanes_cost_the_same_number_of_board_seconds():
    for lane, rate in (("A", PER_A), ("B", PER_B)):
        assert promote_floor(lane) * rate == pytest.approx(BOARD_FLOOR)


def test_lane_b_bar_is_the_fifteen_fold_one_the_flat_floor_hid():
    assert promote_floor("B") / promote_floor("A") == pytest.approx(PER_A / PER_B)
    assert Verdict(1808.2, 1782.2, 26.0, 20.0, 32.0, "B").win is False
    assert Verdict(1808.2, 1754.6, 53.6, 40.0, 66.0, "B").win is True
    assert Verdict(230.1, 228.0, 2.07, 1.12, 2.92, "A").win is True


def test_every_lane_b_verdict_names_its_lane():
    """A lane-B call that forgets the tag silently takes lane A's bar, which is the defect this floor exists to close."""
    calls = re.compile(r"paired_delta\((?:[^()]|\([^()]*\))*\)")
    missed = []
    for f in sorted(os.listdir("taxi")):
        if not f.endswith(".py"):
            continue
        for c in calls.findall(io.open(f"taxi/{f}", encoding="utf-8").read()):
            if "teb" in c and "lane=" not in c:
                missed.append(f"{f}: {c}")
    assert not missed, "lane-B rows judged at lane A's floor: " + "; ".join(missed)
