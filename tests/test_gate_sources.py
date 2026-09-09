"""The gate page is only trustworthy if every figure on it is still the figure its source document holds.

`docs/gate-s4.html` was hand-typed and an independent critique broke four of its numbers. `taxi/gate.py` pins each one to a
verbatim quote; these tests are what make the pin bite.
"""
from __future__ import annotations

import io

import pytest

from taxi.gate import FACTS, GAIN_A, board, check


def test_every_pinned_quote_is_still_in_its_source() -> None:
    check()


@pytest.mark.parametrize("key,src,quote", FACTS)
def test_a_broken_quote_fails_rather_than_ships(key: str, src: str, quote: str) -> None:
    """Mutation test: a quote that no longer matches must raise, not silently render the old number."""
    doc = io.open(src, encoding="utf-8").read()
    assert quote in doc
    assert quote + "​" not in doc, f"{key}: the pin must be an exact substring match, not a fuzzy one"


def test_no_two_facts_share_a_key() -> None:
    keys = [k for k, _s, _q in FACTS]
    assert len(keys) == len(set(keys)), keys


def test_the_gain_the_page_projects_is_the_gain_the_holdout_measured() -> None:
    """GAIN_A is a float in the module and +2.07 is a string in SW1.md; nothing but this test ties them together."""
    holdout = next(q for k, _s, q in FACTS if k == "holdout")
    assert f"+{GAIN_A:.2f}" in holdout, f"gate.py projects {GAIN_A} but SW1's row reads {holdout}"


def test_the_projected_rank_is_better_than_the_scored_one() -> None:
    b = board()
    assert b["after"] < b["v16"] and b["after_rank"] <= b["v16_rank"], b
