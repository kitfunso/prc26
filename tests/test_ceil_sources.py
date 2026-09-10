"""CEIL hand-copies every figure out of another document, so each copy is pinned to a string the source must still contain."""
from __future__ import annotations

import io
import os

import pytest

from taxi.ceil import ITEMS, PAIRS

MS1 = "docs/MS1.md"


@pytest.mark.parametrize("item", ITEMS, ids=[i[4] for i in ITEMS])
def test_source_still_carries_the_quoted_figure(item: tuple) -> None:
    _tier, name, _lane, _sec, src, quote, _note = item
    assert os.path.exists(src), f"{name}: CEIL cites {src}, which does not exist"
    doc = io.open(src, encoding="utf-8").read()
    assert quote in doc, f"{name}: {src} no longer contains {quote!r}, so CEIL's number is stale"


@pytest.mark.parametrize("pair", PAIRS, ids=[p.label for p in PAIRS])
def test_pair_holdout_gain_matches_the_measured_table(pair: tuple) -> None:
    """CEIL's v17/v20 gain was once 0.9585, from a 229.20 nobody measured: SB3 minus LT2's one-decimal published +0.9 s."""
    assert os.path.exists(MS1), f"CEIL's pair table cites {MS1}, which does not exist"
    quote = f"| {pair.label} | {pair.hold:+.4f} |"
    doc = io.open(MS1, encoding="utf-8").read()
    assert quote in doc, f"{MS1} no longer contains {quote!r}, so CEIL's typed holdout gain is stale"


def test_no_gain_is_counted_under_two_tiers() -> None:
    """The BF row was the whole point: its stack contains the net, so it carries BF's increment, not BF's total."""
    assert len({(i[1], i[2]) for i in ITEMS}) == len(ITEMS), "two rows claim the same gain on the same lane"
