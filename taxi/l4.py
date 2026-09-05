"""L4: lane-B band mixture. LIRF's no-type rows get a per-band regime mixture or a per-band line on z_sched; every other row keeps B4."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from taxi.baselines import b4
from taxi.data import CV_FOLDS, HOLDOUT_MONTHS, TRAIN_MONTHS, load
from taxi.l2 import floored
from taxi.l3 import all_rows, by_airport, regime
from taxi.verdict import paired_delta, rmse
from taxi.w1 import verdict_line

K = 2
EDGES = (0.0, 3600.0, 7200.0, 14400.0, 28800.0, 43200.0)
MIN_NOTYP, MIN_BAND = 100, 20
Fit = Callable[[pd.DataFrame, np.ndarray], tuple[float, float]]


def band(z: np.ndarray) -> np.ndarray:
    """0 = (-inf, 0], 1 = (0, 1 h], 2 = (1, 2 h], 3 = (2, 4 h], 4 = (4, 8 h], 5 = (8, 12 h], 6 = (12 h, inf)."""
    return np.searchsorted(np.asarray(EDGES), np.asarray(z, float), "left")


def touched(tr: pd.DataFrame, te: pd.DataFrame) -> np.ndarray:
    n = tr[tr.notyp == 1].groupby("apt").size()
    return (te.notyp.values == 1) & te.apt.isin(n[n >= MIN_NOTYP].index).values


def _mix(t: pd.DataFrame, m: np.ndarray) -> tuple[float, float]:
    """Intercept and slope on z_sched of P(copy) x z + P(day) x airport day mean + P(normal) x band normal mean."""
    reg = regime(t)
    share = np.bincount(reg[m], minlength=3) / m.sum()
    day, norm = t.y.values[reg == 2], t.y.values[m & (reg == 0)]
    mu_day = float(day.mean()) if len(day) else float(t.y.mean())
    mu_norm = float(norm.mean()) if len(norm) else float(t.y.values[reg == 0].mean())
    return share[2] * mu_day + share[0] * mu_norm, float(share[1])


def _ols(t: pd.DataFrame, m: np.ndarray) -> tuple[float, float]:
    a, s = np.linalg.lstsq(np.column_stack([np.ones(int(m.sum())), t.z_sched.values[m]]), t.y.values[m], rcond=None)[0]
    return float(a), float(s)


def per_band(tr: pd.DataFrame, te: pd.DataFrame, fit: Fit) -> np.ndarray:
    """B4 everywhere, then each touched airport's rows get the band's own line a + s z; a thin band uses the airport-wide fit."""
    out, m = b4(tr, te, "B"), touched(tr, te)
    for apt in np.unique(te.apt.values[m]):
        t = tr[(tr.apt == apt) & (tr.notyp == 1)].reset_index(drop=True)
        tb, ix = band(t.z_sched.values), np.flatnonzero(m & (te.apt.values == apt))
        z, zb, whole = te.z_sched.values[ix], band(te.z_sched.values[ix]), fit(t, np.ones(len(t), bool))
        for b in np.unique(zb):
            a, s = fit(t, tb == b) if int((tb == b).sum()) >= MIN_BAND else whole
            out[ix[zb == b]] = a + s * z[zb == b]
    return out


CELLS: dict[str, Fit] = {"L4a": _mix, "L4b": _ols}


def cv(trb: pd.DataFrame) -> tuple[str, list[str]]:
    names = ["B4", *CELLS]
    lines = ["| fold | " + " | ".join(f"{k} RMSE" for k in names) + " |", "|---|" + "---|" * len(names)]
    sq, n = {k: 0.0 for k in names}, 0
    for fold in CV_FOLDS:
        m = trb.month.isin(fold).values
        tr, te, row = trb[~m].reset_index(drop=True), trb[m].reset_index(drop=True), []
        for k in names:
            p = b4(tr, te, "B") if k == "B4" else per_band(tr, te, CELLS[k])
            e = floored(p) - te.y.values
            sq[k], row = sq[k] + float(e @ e), row + [f"{rmse(e):.1f}"]
        n += len(te)
        lines.append(f"| {fold} | " + " | ".join(row) + " |")
        print(lines[-1], flush=True)
    cvr = {k: float(np.sqrt(sq[k] / n)) for k in names}
    best = min(CELLS, key=cvr.get)
    lines.append("\nCV RMSE over the five folds (every cell floored at 0 s): " + ", ".join(f"{k} {v:.1f} s" for k, v in cvr.items()) + f"; best declared cell {best}.")
    print(lines[-1], flush=True)
    return best, lines


def by_band(te: pd.DataFrame, m: np.ndarray, errs: dict[str, np.ndarray]) -> list[str]:
    lines = ["| band | rows | " + " | ".join(f"{k} RMSE" for k in errs) + " |", "|---|---|" + "---|" * len(errs)]
    zb = band(te.z_sched.values)
    for b in range(len(EDGES) + 1):
        ix = np.flatnonzero(m & (zb == b))
        if len(ix):
            lines.append(f"| {b} | {len(ix)} | " + " | ".join(f"{rmse(e[ix]):.0f}" for e in errs.values()) + " |")
    return lines


def main() -> None:
    tr, te = load(TRAIN_MONTHS), load(HOLDOUT_MONTHS)
    trb, teb = (d[~d.has_fpl].reset_index(drop=True) for d in (tr, te))
    best, lines = cv(trb)
    p0, p1 = floored(b4(trb, teb, "B")), floored(per_band(trb, teb, CELLS[best]))
    e0, e1 = p0 - teb.y.values, p1 - teb.y.values
    v, m = paired_delta(e0, e1, teb.day.values), touched(trb, teb)
    jan, jul = teb.month.values == 1, teb.month.values == 7
    row = lambda name, e: f"| {name} | {rmse(e):.1f} | {rmse(e[jan]):.1f} | {rmse(e[jul]):.1f} | {e.min():.0f} | {e.max():.0f} |"
    report = ["# L4: lane-B band mixture, Jan+Jul 2025 development holdout", "",
              f"Train months {TRAIN_MONTHS}; regenerate with `python -m taxi.l4`. Touched rows: no-type rows at airports with at least {MIN_NOTYP} no-type "
              f"training rows ({sorted(teb.apt[m].unique())}); every other row keeps B4. Bands on z_sched at {[int(x) for x in EDGES]} s; a band with fewer "
              f"than {MIN_BAND} training rows uses the airport-wide fit. L4a: P(copy | band) x z_sched + P(day | band) x the airport's day mean + "
              f"P(normal | band) x the band's normal mean (regimes as L3). L4b: per-band least squares y on [1, z_sched]. Every cell and the champion carry "
              f"the floor at 0 s (H22). k = {K}; the CV winner alone meets the holdout.", "",
              "## Month-pair CV (lane-B rows of the fold)", ""] + lines + ["",
              f"## Holdout verdict vs champion B4 with the floor ({len(teb)} rows; {best} differs from it on {int((p1 != p0).sum())} rows)", "",
              "| model | RMSE | Jan | Jul | min error | max error |", "|---|---|---|---|---|---|", row("B4 floored", e0), row(f"{best} floored", e1), "",
              verdict_line(best, "B4", v, k=K), "", "## By airport (reported)", ""] + by_airport(teb, {"B4": e0, best: e1}) + ["",
              f"## Touched rows by band (reported, {int(m.sum())} rows)", ""] + by_band(teb, m, {"B4": e0, best: e1}) + ["",
              f"All rows with B4 (reported, never decides): {all_rows(te, e0)}; with {best}: {all_rows(te, e1)}."]
    text = "\n".join(report) + "\n"
    with open("docs/L4.md", "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
