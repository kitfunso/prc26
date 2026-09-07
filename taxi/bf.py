"""BF: the blend frontier for lane A. Diagnostic, never a verdict (protocol section 2): what a partner model must be to move SB3 by the floor."""
from __future__ import annotations

import numpy as np
import pandas as pd

from taxi.champ import champion_holdout, lane_a
from taxi.l2 import floored
from taxi.sb3 import predict as sb3_predict
from taxi.verdict import rmse

FLOOR = 2.0
LEVELS = (230, 240, 250, 265, 280, 300)


def agreement(e0: np.ndarray, ex: np.ndarray) -> float:
    """Uncentred error correlation, mean(e0 x ex) / (rmse0 x rmsex). RMSE is a second moment, not a variance, so the CENTRED
    correlation makes the closed form below wrong by the product of the two error means; this is the one that keeps it exact."""
    return float(np.mean(e0 * ex) / (rmse(e0) * rmse(ex)))


def blend_rmse(r0: float, rx: float, rho: float, w: float) -> float:
    return float(np.sqrt(w ** 2 * r0 ** 2 + (1 - w) ** 2 * rx ** 2 + 2 * w * (1 - w) * rho * r0 * rx))


def best_blend(e0: np.ndarray, ex: np.ndarray) -> tuple[float, float, float]:
    """The weight minimising the blend's RMSE, read off these very rows, so it upper-bounds any weight a real method could pick."""
    r0, rx = rmse(e0), rmse(ex)
    rho = agreement(e0, ex)
    w = (rx ** 2 - rho * r0 * rx) / (r0 ** 2 + rx ** 2 - 2 * rho * r0 * rx)
    return w, blend_rmse(r0, rx, rho, w), rho


def needed_rho(r0: float, rx: float, gain: float = FLOOR) -> float:
    """Highest error correlation at which a partner of this level still reaches the gain, by bisection on the closed form above."""
    lo, hi = 0.0, 0.9999
    for _ in range(70):
        m = (lo + hi) / 2
        w = (rx ** 2 - m * r0 * rx) / (r0 ** 2 + rx ** 2 - 2 * m * r0 * rx)
        lo, hi = (m, hi) if r0 - blend_rmse(r0, rx, m, w) >= gain else (lo, m)
    return lo


def main() -> None:
    tra, tea = lane_a()
    y = tea.y.values
    cached = lambda p: floored(pd.read_parquet(p).set_index("id").pred.reindex(tea.id.values).values)  # noqa: E731
    e0 = sb3_predict(tra, tea) - y
    r0 = rmse(e0)
    partners = (("MC2, CatBoost, the champion's own family", floored(champion_holdout("MC2", tra, tea))),
                ("D1, LightGBM, a different library", cached("data/cache/d1_holdout.parquet")),
                ("M3, TabPFN per airport, a different architecture", cached("data/cache/m3_holdout.parquet")))
    rows, gains, rhos = [], [], []
    for name, p in partners:
        w, b, rho = best_blend(e0, p - y)
        gains.append(r0 - b)
        rhos.append(rho)
        rows.append(f"| {name} | {rmse(p - y):.2f} | {rho:.4f} | {b:.2f} | {w:.3f} | {r0 - b:+.2f} |")
    front = [f"| {rx} | {needed_rho(r0, float(rx)):.3f} |" for rx in LEVELS]
    out = ["# BF: what a partner model must be to move lane A, Jan + Jul 2025 development holdout", "",
           f"Diagnostic, never a verdict, no cell declared. Regenerate with `python -m taxi.bf`. Lane-A champion SB3, the five-seed D2 bag, at {r0:.2f} s on "
           f"{len(tea):,} rows. For two predictors the blend RMSE is closed form in their two levels and their error correlation, so the "
           "best weight and the gain it buys need no fit. Agreement is the UNCENTRED error correlation, mean(e0 x ex) over the two RMSEs, which is the form that keeps the identity exact when the errors are not zero mean. The weight below is read off the holdout itself and is therefore unavailable in "
           "practice: it upper-bounds what any out-of-fold choice could reach.", "",
           "## Every lane-A model this campaign has cached, blended with the champion", "",
           "| partner | its RMSE (s) | error agreement with SB3 | best blend (s) | its weight | gain (s) |",
           "|---|---|---|---|---|---|", *rows, "",
           f"The three partners span the whole model-class axis, from the champion's own library to a transformer, and the gain rises "
           f"{gains[0]:+.2f} to {gains[-1]:+.2f} s against a {FLOOR:.0f} s floor.", "",
           f"## The frontier: error agreement a partner needs for a {FLOOR:.0f} s gain", "",
           "| partner RMSE (s) | agreement it must be under |", "|---|---|", *front, "",
           f"The least agreeing partner is TabPFN at {rhos[-1]:.3f}, and at its {rmse(partners[-1][1] - y):.0f} s level the frontier asks for "
           f"{needed_rho(r0, rmse(partners[-1][1] - y)):.3f}. A partner at the champion's own level would have to stay under "
           f"{needed_rho(r0, r0):.3f}, and the two GBM libraries measured here read {rhos[0]:.4f} and {rhos[1]:.4f}. "
           "Reading: a further model class does not reach the floor by blending, whatever library it comes from. It reaches it only by "
           "being better than SB3 outright, which is a different question and needs no blend.", ""]
    text = "\n".join(out)
    with open("docs/BF.md", "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
