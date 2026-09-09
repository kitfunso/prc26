"""RW1: training window and recency weighting for lane A, judged on the Dec-2025 forward fold. Reported, never decides (protocol section 4)."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from taxi.champ import COLS_D1, final_rounds
from taxi.data import load
from taxi.mc1 import predict_cb, stages_cb
from taxi.verdict import rmse
from taxi.weather import join, load_metar

FIT_MONTHS, JUDGE = tuple(range(1, 12)), (12,)
WINDOWS, HALFLIVES = (3, 6, 9, 11), (3.0, 6.0, 12.0)
VAL_FRAC = 0.10  # the last tenth of fit days, by day, early-stops every arm alike; it abuts December the way December abuts the ranking file


def arms() -> list[tuple[str, tuple[int, ...], float | None]]:
    """Name, months kept, weight half-life in months. The 11-month unweighted arm is MC2 itself and anchors the table."""
    return [(f"W{n}", tuple(range(12 - n, 12)), None) for n in WINDOWS] + \
           [(f"H{h:g}", FIT_MONTHS, h) for h in HALFLIVES]


def weights(df: pd.DataFrame, hl: float | None) -> np.ndarray | None:
    if hl is None:
        return None
    w = 0.5 ** ((12 - df.month.values) / hl)
    return w / w.mean()  # mean 1 keeps l2_leaf_reg comparable to the unweighted fit


def split(fit: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    days = np.sort(fit.day.unique())
    cut = days[int(len(days) * (1 - VAL_FRAC))]
    return fit.day.values < cut, fit.day.values >= cut


def run_arm(fit: pd.DataFrame, dec: pd.DataFrame, months: tuple[int, ...], hl: float | None,
            rounds: tuple[int, int, int] | None) -> tuple[float, int, str]:
    f = fit[fit.month.isin(months)].reset_index(drop=True)
    if rounds is not None:
        models = stages_cb(f, iters=rounds, cols=COLS_D1, w=weights(f, hl))
        trees = "fixed"
    else:
        tr, va = split(f)
        models = stages_cb(f[tr].reset_index(drop=True), va=f[va].reset_index(drop=True), cols=COLS_D1,
                           w=weights(f[tr].reset_index(drop=True), hl))
        trees = str(tuple(b.tree_count_ for b in models))
    return rmse(predict_cb(models, dec, COLS_D1)[0] - dec.y.values), len(f), trees


def main() -> None:
    metar = load_metar()
    fit, dec = (join(load(m), metar) for m in (FIT_MONTHS, JUDGE))
    fit, dec = (x[x.has_fpl].reset_index(drop=True) for x in (fit, dec))
    rounds = final_rounds("MC2", "docs/MC2.md")
    lines = ["| arm | months | half-life | fit rows | trees | Dec RMSE (s) | vs MC2-es (s) | fit s |", "|---|---|---|---|---|---|---|---|"]
    todo = [("MC2-fixed", FIT_MONTHS, None, rounds), ("MC2-es", FIT_MONTHS, None, None)] + [(n, m, h, None) for n, m, h in arms()]
    base = None
    for name, months, hl, rd in todo:
        t0 = time.time()
        r, n, trees = run_arm(fit, dec, months, hl, rd)
        base = r if name == "MC2-es" else base
        gap = "" if base is None or name == "MC2-es" else f"{base - r:+.1f}"
        lines.append(f"| {name} | {months[0]}-{months[-1]} | {'-' if hl is None else f'{hl:g} mo'} | {n} | {trees} | {r:.1f} | {gap} | {time.time() - t0:.0f} |")
        print(lines[-1], flush=True)
    text = "\n".join([
        "# RW1: training window and recency weighting, Dec-2025 forward fold", "",
        f"Regenerate with `python -m taxi.rw1`. MC2's cell and its {len(COLS_D1)} columns throughout; train on 2025 months "
        f"{FIT_MONTHS[0]}-{FIT_MONTHS[-1]}, judge {len(dec)} December lane-A rows. `MC2-fixed` refits at MC2's recorded rounds {rounds} and "
        f"reproduces `docs/FORWARD.md`; every other arm early-stops on the last {VAL_FRAC:.0%} of fit days and is compared to `MC2-es`, which "
        "early-stops the same way on the same rows. k = 7 declared arms (protocol section 4, RW1).", "",
        *lines, "",
        "This fold cannot promote a champion. The holdout is Jan + Jul 2025, interior to the training year, so a recency window has no meaning "
        "against it, while the submission trains on all of 2025 and predicts 2026. Section 3 declares the forward fold reported, never deciding, "
        "so a winning arm is a gate question about how `taxi/s2.py` builds the submission, not a promotion.", "",
        "Hazards recorded at declaration: December is one month and holds Christmas, so this is one seasonal draw; and `docs/HB.md` arm D "
        "measured seasonal weighting at -0.9444 s, a loss, though it weighted by calendar similarity rather than by time distance.", ""]) + "\n"
    with open("docs/RW1.md", "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
