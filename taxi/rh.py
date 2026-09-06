"""RH: does a recent history of the REALISED clock buy anything MC2 does not already have?

Diagnostic. No cell declared, no champion moves, no verdict decided. Every feature in the build is a
plan (z_aobt, z_sched) or a count (queue, occ, dep_before); none is a realised outcome of a past flight.
Regenerate with `python -m taxi.rh`.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from taxi.champ import lane_a
from taxi.data import month_file

RANKING = "data/ranking.parquet"
HOLDOUT_MONTHS = (1, 7)
NBINS = 20  # one fixed cell: this prices a channel, it must not become an undeclared search.


def avail() -> dict[str, int]:
    """The channel is only real if the ranking set carries it; a departure label there would be the answer itself."""
    q = f"""SELECT PHASE_mvt p, count(*) n, count(TAXITIME_SEC_mvt) lab, count(BLOCK_TIME_UTC_mvt) blk
            FROM read_parquet('{RANKING}') GROUP BY 1"""
    d = duckdb.sql(q).df().set_index("p")
    return {f"{p}_{c}": int(d.loc[p, c]) for p in d.index for c in ("n", "lab", "blk")}


def arrivals() -> pd.DataFrame:
    paths = [month_file(m) for m in HOLDOUT_MONTHS]
    return duckdb.sql(f"""SELECT ADES_mvt AS apt, epoch(BLOCK_TIME_UTC_mvt) AS tb, TAXITIME_SEC_mvt::DOUBLE AS ti
                          FROM read_parquet({paths}) WHERE PHASE_mvt='ARR'
                            AND BLOCK_TIME_UTC_mvt IS NOT NULL AND TAXITIME_SEC_mvt IS NOT NULL
                          ORDER BY apt, tb""").df()


def _mean_before(t: np.ndarray, v: np.ndarray, anc: np.ndarray, k: int | None, win: float | None) -> np.ndarray:
    """Mean of v over events already complete at anc. searchsorted left is what makes it point-in-time."""
    cs = np.concatenate([[0.0], np.cumsum(v)])
    pos = np.searchsorted(t, anc, "left")
    lo = np.maximum(pos - k, 0) if k else np.searchsorted(t, anc - win, "left")
    return np.where(pos - lo >= 3, (cs[pos] - cs[lo]) / np.maximum(pos - lo, 1), np.nan)


def rec_c(df: pd.DataFrame, k: int = 10, by: tuple[str, ...] = ("apt",)) -> np.ndarray:
    """Recent realised clock offset c = y - z_aobt of past DEPARTURES: the target's own quantity, one flight back."""
    anc, out = df.t_off.values - df.z_anchor.values, np.full(len(df), np.nan)
    c, t_off = df.y.values - df.z_aobt.values, df.t_off.values
    for _, ix in df.groupby(list(by), observed=True).indices.items():
        o = np.argsort(t_off[ix])
        out[ix] = _mean_before(t_off[ix][o], c[ix][o], anc[ix], k, None)
    return out


def rec_in(df: pd.DataFrame, arr: pd.DataFrame, k: int | None = None, win: float | None = None) -> np.ndarray:
    """Recent realised taxi-IN of arrivals already on stand: the same surface, and labelled on the ranking set."""
    anc, out = df.t_off.values - df.z_anchor.values, np.full(len(df), np.nan)
    for a, g in arr.groupby("apt", observed=True):
        ix = np.flatnonzero(df.apt.values == a)
        if len(ix):
            out[ix] = _mean_before(g.tb.values, g.ti.values, anc[ix], k, win)
    return out


def gain(v: np.ndarray, pred: np.ndarray, y: np.ndarray, res: np.ndarray, mon: np.ndarray) -> float:
    """Out of fold across the two holdout months: bin the residual on one month, apply the bin's mean to the other."""
    p = pred.copy()
    for fit, app in ((mon == 1, mon == 7), (mon == 7, mon == 1)):
        ok = fit & ~np.isnan(v)
        q = np.quantile(v[ok], np.linspace(0, 1, NBINS + 1))
        q[0], q[-1] = -np.inf, np.inf
        b = np.digitize(v, q[1:-1])
        adj = np.array([res[ok & (b == j)].mean() if (ok & (b == j)).sum() > 30 else 0.0 for j in range(NBINS)])
        m = app & ~np.isnan(v)
        p[m] -= adj[b[m]]
    return float(np.sqrt(np.mean((pred - y) ** 2)) - np.sqrt(np.mean((p - y) ** 2)))


def main() -> None:
    a = avail()
    _, tea = lane_a()
    tea = tea.reset_index(drop=True)
    arr = arrivals()
    mc2 = pd.read_parquet("data/cache/mc2_holdout.parquet")
    pred, y, mon = mc2.pred.values, tea.y.values, tea.month.values
    res = pred - y
    base = float(np.sqrt(np.mean(res ** 2)))

    def row(name: str, v: np.ndarray) -> str:
        """The ceiling is what a perfect additive use of the channel could return, so a negative bin gain is not the only evidence."""
        ok = ~np.isnan(v)
        r = np.corrcoef(v[ok], res[ok])[0, 1]
        ceil = base * (1 - np.sqrt(max(1 - r * r, 0.0))) * ok.mean()
        return (f"| {name} | {100 * ok.mean():.1f}% | {np.corrcoef(v[ok], y[ok])[0, 1]:+.4f} | {r:+.4f} | "
                f"{ceil:.3f} | {gain(v, pred, y, res, mon):+.3f} |")

    dep = [row(n, rec_c(tea, **kw)) for n, kw in
           (("last 10 departures, airport", dict(k=10)), ("last 30, airport", dict(k=30)),
            ("last 100, airport", dict(k=100)), ("last 10, airport and airline", dict(k=10, by=("apt", "airline"))),
            ("last 10, airport and runway", dict(k=10, by=("apt", "runway"))))]
    inb = [row(n, rec_in(tea, arr, **kw)) for n, kw in
           (("last 10 arrivals", dict(k=10)), ("last 30 arrivals", dict(k=30)), ("last 60 arrivals", dict(k=60)),
            ("arrivals on stand in the previous 30 min", dict(win=1800.0)),
            ("arrivals on stand in the previous 60 min", dict(win=3600.0)))]
    best_in = max(float(r.split("|")[6]) for r in inb)
    ceil_in = max(float(r.split("|")[5]) for r in inb)

    hdr = ["| channel | populated | corr with y | corr with the MC2 residual | linear ceiling (s) | out-of-fold gain (s) |",
           "|---|---|---|---|---|---|"]
    out = [
        "# RH: a recent history of the realised clock, in both directions",
        "",
        "Regenerate with `python -m taxi.rh`. Diagnostic: no cell declared, no champion moves, no verdict decided.",
        f"MC2 holdout RMSE {base:.4f} s on {len(tea):,} lane-A rows, against the 2.0 s improvement floor.",
        "",
        "## 1. Why the question is open",
        "",
        "The lane-A target is `c = AOBT_3 - BLOCK_TIME`, a disagreement between the flight-plan clock and the movement",
        "clock. A clock offset should be persistent and airport-specific, so the natural missing column is what that",
        "offset recently was. `rec_taxi` looks like it, but averages the PLANNED taxi `z_aobt` of the last 10 takeoffs;",
        "no feature in the build is a realised outcome of a past flight. The construction below is point-in-time by the",
        "same argument `rec_taxi` already relies on: `searchsorted(t, anchor, 'left')` admits only events already complete.",
        "",
        "## 2. Availability decides the departure direction before signal does",
        "",
        f"| ranking.parquet | rows | labelled | with a block time |",
        "|---|---|---|---|",
        f"| departures | {a['DEP_n']:,} | {a['DEP_lab']:,} | {a['DEP_blk']:,} |",
        f"| arrivals | {a['ARR_n']:,} | {a['ARR_lab']:,} | {a['ARR_blk']:,} |",
        "",
        f"The ranking set carries no departure label at all, so a recent realised DEPARTURE offset cannot be computed",
        "there: the preceding departures are exactly the rows we are asked to predict. It is measured below anyway,",
        "because a large holdout gain would have been worth finding a proxy for, and because the measurement is what",
        "makes the closure a fact rather than an assumption.",
        "",
        f"The same table opens the other direction. Every one of the {a['ARR_lab']:,} arrivals in the ranking set carries",
        "its realised taxi-in and its block time. That is a labelled measurement of the same surface, at the same",
        "airports, in the same window, and no feature reads it: arrivals enter the build only as event times, through",
        "`LANDINGS` for the counts and `ONBLOCK` for stand occupancy.",
        "",
        "## 3. Both directions, against MC2's cached holdout residuals",
        "",
        "Realised departure offset (not available on the ranking set):",
        "",
        *hdr, *dep, "",
        "Realised arrival taxi-in (available on the ranking set):",
        "",
        *hdr, *inb, "",
        "## 4. Answer: the family is closed, for two different reasons",
        "",
        f"The departure direction has the stronger raw signal of the two and still does not reach the floor. Its best",
        f"out-of-fold correction is {max(float(r.split(chr(124))[6]) for r in dep):+.3f} s, about a quarter of the 2.0 s floor, and its linear ceiling is the",
        "same size: a residual correlation of -0.078 caps an additive correction at 0.70 s by arithmetic. So the channel",
        "would have been parked under the floor even if the ranking set had carried the labels, and the availability",
        "fact makes it moot rather than being the only thing against it.",
        "",
        f"The arrival direction is available and is the one that matters, and it is flat: it correlates with the label",
        f"but not with the residual, every out-of-fold correction is negative, the best of them {best_in:+.3f} s, and its linear",
        f"ceiling is {ceil_in:.3f} s, so the closure does not rest on the sign of a bin correction. MC2's 49",
        "columns already span it. `queue`, `occ`, `dep_before`, `arr_anchor`, the hour and the airport describe the same",
        "congestion the arrival taxi-in measures, and they describe it without the noise of averaging ten arrivals.",
        "",
        "Consequence for the campaign: do not re-ask for a recent-realised-history feature in either direction, and do",
        "not treat the labelled 2026 arrivals as an unused channel. They are unused as a LABEL and spanned as a SIGNAL.",
        "",
    ]
    text = "\n".join(out) + "\n"
    with open("docs/RH.md", "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
