"""CEIL hand-copies every figure out of another document, so each copy is pinned to a string the source must still contain."""
from __future__ import annotations

import io
import os

import pytest

from taxi.ceil import ITEMS


@pytest.mark.parametrize("item", ITEMS, ids=[i[4] for i in ITEMS])
def test_source_still_carries_the_quoted_figure(item: tuple) -> None:
    _tier, name, _lane, _sec, src, quote, _note = item
    assert os.path.exists(src), f"{name}: CEIL cites {src}, which does not exist"
    doc = io.open(src, encoding="utf-8").read()
    assert quote in doc, f"{name}: {src} no longer contains {quote!r}, so CEIL's number is stale"


def test_no_gain_is_counted_under_two_tiers() -> None:
    """The BF row was the whole point: its stack contains the net, so it carries BF's increment, not BF's total."""
    assert len({(i[1], i[2]) for i in ITEMS}) == len(ITEMS), "two rows claim the same gain on the same lane"
