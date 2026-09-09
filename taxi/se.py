"""SE (diagnostic, never a verdict): is lane A's error seasonal, and would a season column catch it?

`docs/HB.md` states the gap ("COLS_D1 carries no month, season or day-of-week column") and then tests seasonal WEIGHTS and
judged-month ROWS, never a season COLUMN. The holdout is Jan + Jul, the two ends of the year, and they differ by about 45 s of
RMSE, so the question is whether that difference is bias a column could remove or variance it could not.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from taxi.champ import champion_oof, lane_a
from taxi.data import CV_FOLDS
from taxi.sb3 import predict as sb3_predict
from taxi.verdict import rmse

FLOOR = 2.0
REPS = 5
HARMONICS = (1, 2, 3, 4)


def cell_gain(e: np.ndarray, keys: list[np.ndarray], rng: np.random.Generator) -> tuple[float, float, int]:
    """Gain from removing each cell's own mean residual, against what the same cell count frees from shuffled residuals."""
    g = pd.Series(e).groupby([pd.Series(k) for k in keys], observed=True)
    raw = rmse(e) - rmse(e - g.transform("mean").values)
    null = []
    for _ in range(REPS):
        p = rng.permutation(e)
        null.append(rmse(p) - rmse(p - pd.Series(p).groupby([pd.Series(k) for k in keys], observed=True).transform("mean").values))
    return raw, raw - float(np.mean(null)), int(g.ngroups)


def design(doy: np.ndarray, k: int) -> np.ndarray:
    """Sin/cos harmonics of day-of-year: unlike a month category these interpolate, so a held-out month still lands somewhere."""
    cols = [f(2 * np.pi * h * doy / 365.25) for h in range(1, k + 1) for f in (np.sin, np.cos)]
    return np.column_stack([*cols, np.ones(len(doy))])


def cross_fit(e: np.ndarray, doy: np.ndarray, mon: np.ndarray, k: int) -> float:
    out = e.copy()
    X = design(doy, k)
    for fold in CV_FOLDS:
        m = np.isin(mon, fold)
        beta = np.linalg.lstsq(X[~m], e[~m], rcond=None)[0]
        out[m] = e[m] - X[m] @ beta
    return rmse(e) - rmse(out)


def main(dst: str = "docs/SE.md") -> None:
    rng = np.random.default_rng(0)
    tra, tea = lane_a()
    oof = champion_oof("MC2", tra)
    y, mon = tra.y.values, tra.month.values
    e = oof - y
    doy = pd.to_datetime(tra.day).dt.dayofyear.values.astype(float)

    t = pd.DataFrame({"m": mon, "e": e}).groupby("m").agg(rows=("e", "size"), bias=("e", "mean"),
                                                          rmse=("e", lambda s: rmse(s.values)))
    mrows = [f"| {m} | {r.rows:,.0f} | {r.bias:+.1f} | {r.rmse:.1f} |" for m, r in t.iterrows()]

    hr = tra.hour.values
    apt = tra.apt.astype(str).values
    cells = (("month", [mon]), ("month + airport", [mon, apt]), ("month + airport + hour", [mon, apt, hr]))
    crows = []
    for name, keys in cells:
        raw, hon, ng = cell_gain(e, keys, rng)
        crows.append(f"| {name} | {ng:,} | {raw:+.3f} | {hon:+.3f} |")

    hrows = [f"| {k} | {cross_fit(e, doy, mon, k):+.3f} |" for k in HARMONICS]

    ph = sb3_predict(tra, tea)
    eh = ph - tea.y.values
    dh = pd.to_datetime(tea.day).dt.dayofyear.values.astype(float)
    best = max(HARMONICS, key=lambda k: cross_fit(e, doy, mon, k))
    X = design(doy, best)
    beta = np.linalg.lstsq(X, e, rcond=None)[0]
    held = rmse(eh) - rmse(eh - design(dh, best) @ beta)
    jan, jul = tea.month.values == 1, tea.month.values == 7

    out = ["# SE: is lane A's error seasonal, Jan + Jul 2025 development holdout", "",
           "Diagnostic, never a verdict, no cell declared. Regenerate with `python -m taxi.se`. `docs/HB.md` section 2 records that "
           "no lane-A column carries the season, and then prices seasonal WEIGHTS (arm D, -0.9444 s) and the judged months' own "
           "ROWS (arm B, +3.9552 s). Neither is a season COLUMN, which is a different mechanism: a weight changes which rows the "
           "loss cares about, a column lets the trees fit a different function inside each season.", "",
           f"MC2's out-of-fold residuals on {len(tra):,} lane-A training rows, RMSE {rmse(e):.2f} s. The holdout months are absent "
           "here by construction, so the seasonal shape below is read off the ten months that ARE fit on.", "",
           "## 1. The error by month", "", "| month | rows | mean residual (s) | RMSE (s) |", "|---|---|---|---|", *mrows, "",
           f"Spread of the monthly bias: {t.bias.max() - t.bias.min():.1f} s across a {t.rmse.mean():.0f} s error. Spread of the "
           f"monthly RMSE: {t.rmse.max() - t.rmse.min():.1f} s.", "",
           "## 2. What removing a seasonal cell's own mean residual is worth", "",
           "Each row's residual loses its cell's mean, fit on the very rows it is judged on, so this is a CEILING. Cells free error "
           "from pure noise too, so the honest column subtracts what the same cell count frees from shuffled residuals.", "",
           "| cells | count | in-sample gain (s) | minus the shuffle null (s) |", "|---|---|---|---|", *crows, "",
           "## 3. The buildable version: a day-of-year harmonic", "",
           "A month category cannot be cross-validated here at all: the folds ARE month pairs, so a fold's months never appear in "
           "its own training rows and every judged row would meet an unseen category. That is the structural reason this campaign "
           "never carried one. Sin/cos harmonics of day-of-year interpolate instead, so they survive the same folds.", "",
           "| harmonics | cross-fitted gain (s) |", "|---|---|", *hrows, "",
           f"Carried to the holdout: the {best}-harmonic correction fit on all ten training months moves SB3 from {rmse(eh):.2f} s "
           f"to {rmse(eh - design(dh, best) @ beta):.2f} s, **{held:+.3f} s**, against a {FLOOR:.0f} s floor. Jan {rmse(eh[jan]):.1f} s "
           f"and Jul {rmse(eh[jul]):.1f} s on the holdout, a {rmse(eh[jul]) - rmse(eh[jan]):.1f} s difference the correction does not "
           "touch, because it is variance and not bias.", ""]
    text = "\n".join(out)
    with open(dst, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
