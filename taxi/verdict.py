"""Paired day-block bootstrap behind every holdout verdict (protocol section 3)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FLOOR = 2.0


@dataclass(frozen=True)
class Verdict:
    champion: float
    challenger: float
    delta: float
    lo: float
    hi: float

    @property
    def win(self) -> bool:
        return self.lo > 0 and self.delta >= FLOOR

    def line(self) -> str:
        tag = "WIN" if self.win else "tie or lose"
        return f"improvement {self.delta:+.1f} s [{self.lo:+.1f}, {self.hi:+.1f}], floor {FLOOR:.0f} s: {tag}"


def rmse(err: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(err))))


def paired_delta(err_champion, err_challenger, days, reps: int = 2000, seed: int = 0, block: int = 1) -> Verdict:
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
    return Verdict(rmse(a), rmse(b), rmse(a) - rmse(b), float(lo), float(hi))


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
