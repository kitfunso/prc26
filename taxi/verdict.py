"""Paired day-block bootstrap behind every holdout verdict (protocol section 3)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from taxi.ceil import PER_A, PER_B

FLOOR = 2.0  # parsimony tolerance in the metric's OWN seconds: "within this of the best cell, prefer the simpler one"
BOARD_FLOOR = 2.0 * PER_A  # what promotion costs, in BOARD seconds, pinned so lane A's bar stays exactly 2.0 holdout s


def promote_floor(lane: str) -> float:
    """Holdout seconds a challenger on `lane` must gain to be worth `BOARD_FLOOR` on the board.

    A holdout second is not one thing. `taxi.ceil` measures lane A at 0.8723 board seconds per holdout second off the controlled
    v15/v16 pair, and puts lane B at 0.0581. One flat holdout floor therefore asked lane A for 1.74 board seconds and lane B for
    0.10, so the two lanes competed at a 15x handicap in the only currency the prize is decided in. Lane B's rate is a FLOOR, not
    a measurement (`docs/BL.md` section 5: no controlled lane-B pair has ever been scored), so dividing by it gives the most
    demanding bar available. If the pooled derivation understates lane B the way it understated lane A, by 1.87x, the true lane-B
    bar is nearer 16 s than 30 s; that stays unmeasured until a controlled lane-B pair is scored.
    """
    return BOARD_FLOOR / (PER_A if lane == "A" else PER_B)


@dataclass(frozen=True)
class Verdict:
    champion: float
    challenger: float
    delta: float
    lo: float
    hi: float
    lane: str = "A"

    @property
    def floor(self) -> float:
        return promote_floor(self.lane)

    @property
    def win(self) -> bool:
        return self.lo > 0 and self.delta >= self.floor

    def line(self) -> str:
        tag = "WIN" if self.win else "tie or lose"
        return f"improvement {self.delta:+.1f} s [{self.lo:+.1f}, {self.hi:+.1f}], floor {self.floor:.0f} s: {tag}"


def rmse(err: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(err))))


def score(pred: np.ndarray, te) -> str:
    """A progress line's RMSE fragment, empty when the frame carries no label: the ranking file has none (F36, F39)."""
    return f", RMSE {rmse(pred - te.y.values):.2f}" if "y" in te.columns else ""


def paired_delta(err_champion, err_challenger, days, reps: int = 2000, seed: int = 0, block: int = 1, lane: str = "A") -> Verdict:
    """Champion RMSE minus challenger RMSE on the same rows, resampling whole days; positive favours the challenger."""
    a = np.asarray(err_champion, dtype=float)
    b = np.asarray(err_challenger, dtype=float)
    _, inv = np.unique(np.asarray(days), return_inverse=True)
    grp = inv // block
    nb = int(grp.max()) + 1
    n = np.bincount(grp, minlength=nb).astype(float)
    sse_a = np.bincount(grp, weights=a * a, minlength=nb)
    sse_b = np.bincount(grp, weights=b * b, minlength=nb)
    pick = np.random.default_rng(seed).integers(0, nb, size=(reps, nb))
    tot = n[pick].sum(axis=1)
    d = np.sqrt(sse_a[pick].sum(axis=1) / tot) - np.sqrt(sse_b[pick].sum(axis=1) / tot)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return Verdict(rmse(a), rmse(b), rmse(a) - rmse(b), float(lo), float(hi), lane)


def level_ci(err, days, reps: int = 2000, seed: int = 0) -> tuple[float, float, float, float]:
    """Day-block bootstrap of an RMSE LEVEL (not a paired delta): the point value, its bootstrap SD, and the 2.5-97.5 range.

    Levels are what the board reports, and the composite's level is carried by a handful of extreme rows, so its sampling spread
    is far wider than any paired verdict on the same rows.
    """
    e = np.asarray(err, dtype=float)
    _, grp = np.unique(np.asarray(days), return_inverse=True)
    nb = int(grp.max()) + 1
    n = np.bincount(grp, minlength=nb).astype(float)
    sse = np.bincount(grp, weights=e * e, minlength=nb)
    pick = np.random.default_rng(seed).integers(0, nb, size=(reps, nb))
    r = np.sqrt(sse[pick].sum(axis=1) / n[pick].sum(axis=1))
    lo, hi = np.percentile(r, [2.5, 97.5])
    return rmse(e), float(r.std(ddof=1)), float(lo), float(hi)
