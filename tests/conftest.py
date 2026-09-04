import os

import pytest

from taxi.data import DATA, features, month_file


@pytest.fixture(scope="session")
def feb_features():
    if not os.path.isdir(DATA):
        pytest.skip("data/ not present")
    return features([month_file(2)])
