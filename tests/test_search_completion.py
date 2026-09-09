"""The two rules a declared search must satisfy: its budget must be arithmetically reachable, and a short search must say so."""
import io
import os

import pytest

from taxi.searches import SEARCHES, SHORT_DOC, SHORT_MARK, achieved, clock_s, rate_s

IDS = [s.name for s in SEARCHES]


@pytest.mark.parametrize("s", SEARCHES, ids=IDS)
def test_the_declared_trial_count_fits_inside_the_declared_clock(s):
    """Rule A. O2 declared 40 trials or 10 h at a measured 40 min a trial, so its clock always ended it near trial 15.

    The rate comes from `rate_s`, which prefers the study's own wall clock over the hand-written declaration. Checking
    the declaration against itself is how this rule passed O2 twice: the number it was given was the one that was wrong.
    """
    r, src = rate_s(s)
    need = s.k * r
    assert need <= clock_s(s), (f"{s.name}: {s.k} trials x {r / 60:.0f} min ({src}) needs {need / 3600:.1f} h "
                                f"against a declared clock of {clock_s(s) / 3600:.1f} h")


@pytest.mark.parametrize("s", SEARCHES, ids=IDS)
def test_a_search_under_its_declared_k_is_named_in_the_notdone_plan(s):
    """Rule B. A search that stopped short is a NOT-DONE row with its partial k, never a silent 'queued'."""
    n = achieved(s)
    if n >= s.k:
        return
    assert os.path.exists(SHORT_DOC), f"{SHORT_DOC} is missing"
    text = io.open(SHORT_DOC, encoding="utf-8").read()
    assert SHORT_MARK in text, f"{SHORT_DOC} has no '{SHORT_MARK}' section"
    assert f"| {s.name} |" in text.split(SHORT_MARK, 1)[1], (
        f"{s.name} stands at {n} of its declared {s.k} trials and has no row under '{SHORT_MARK}' in {SHORT_DOC}")
