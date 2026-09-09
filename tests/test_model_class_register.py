"""The register is only a checklist if the library is the thing enumerated, not the register's own memory."""
from __future__ import annotations

from sklearn.utils import all_estimators

from taxi.register import ROWS


def installed() -> set[str]:
    return {n for n, _ in all_estimators("regressor")}


def test_every_installed_regressor_is_classified() -> None:
    covered = {n for r in ROWS for n in r.covers}
    missing = installed() - covered
    assert not missing, f"scikit-learn ships regressors no register row owns: {sorted(missing)}"


def test_no_row_names_an_absent_estimator() -> None:
    covered = {n for r in ROWS for n in r.covers}
    stale = covered - installed()
    assert not stale, f"register rows name estimators this sklearn does not have: {sorted(stale)}"


def test_no_estimator_is_claimed_twice() -> None:
    seen: dict[str, str] = {}
    dupes = []
    for r in ROWS:
        for n in r.covers:
            if n in seen:
                dupes.append(f"{n}: {seen[n]} and {r.family}")
            seen[n] = r.family
    assert not dupes, f"an estimator belongs to exactly one family: {dupes}"


def test_every_row_carries_a_status() -> None:
    assert all(r.status.strip() for r in ROWS), [r.family for r in ROWS if not r.status.strip()]


def test_an_empty_row_names_no_instance() -> None:
    """An EMPTY row with a lane in it is the exact confusion the register exists to prevent."""
    bad = [r.family for r in ROWS if r.status.startswith("EMPTY") and r.instance]
    assert not bad, f"rows marked EMPTY but carrying an instance: {bad}"


def test_coverage_doc_carries_the_generated_table() -> None:
    """The doc used to hold its own copy of this table, which is the drift the register exists to remove."""
    import io

    from taxi.register import table

    doc = io.open("docs/COVERAGE.md", encoding="utf-8").read()
    assert table() in doc, "docs/COVERAGE.md is out of date: regenerate its register table from taxi/register.py"
