"""Session-scoped fixtures shared by the R5 audit tests: raw departures and the built feature frame,
each loaded once per pytest run instead of once per test file."""
from __future__ import annotations

import pytest

from data import departures, load_raw
from features import build


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: touches the full raw/feature frame or rebuilds it; skip with -m 'not slow'")


@pytest.fixture(scope="session")
def dep():
    """All departure rows (train + rank), raw movement columns intact (e.g. BLOCK_TIME_UTC_mvt, TAXITIME_SEC_mvt)."""
    return departures(load_raw())


@pytest.fixture(scope="session")
def feat():
    """The cached feature frame from features.build(): train + rank, every CAT/NUM/DAY/OCC/WX/ARR/WZ/SNOW column."""
    return build()
