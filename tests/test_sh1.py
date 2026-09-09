"""The one way SH1 can lie: judging on the month it nominated from. A model built to be great in January and awful in July catches it."""
import numpy as np

from taxi.sh1 import one_direction


def test_the_judged_score_comes_from_the_month_the_nomination_never_saw():
    month = np.repeat([1, 7], 100)
    y = np.zeros(200)
    fam = {"champ": np.full(200, 10.0),
           "jan_specialist": np.where(month == 1, 1.0, 100.0)}  # wins January, disastrous in July

    line, _ = one_direction(fam, y, month, "champ", nom=1, jud=7)
    assert "jan_specialist" in line, "January must nominate the model that wins January"
    assert "100.00" in line, "the judged number must be July's 100, not January's 1"
    assert "+90.00" in line, "haircut = honest 100 minus as-run 10"

    back, _ = one_direction(fam, y, month, "champ", nom=7, jud=1)
    assert "champ" in back and "10.00" in back, "July nominates the champion and January judges it at 10"
