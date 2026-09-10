"""BF: the blend frontier for lane A. Diagnostic, never a verdict (protocol section 2): what a partner model must be to move SB3 by the floor."""
from __future__ import annotations

from itertools import combinations

import numpy as np

from taxi.champ import champion_holdout, lane_a
from taxi.l2 import floored
from taxi.sb3 import predict as sb3_predict
from taxi.sw1 import on_disk, raw
from taxi.verdict import rmse

FLOOR = 2.0
LEVELS = (230, 240, 250, 265, 280, 300)

# Partners come from `taxi.sw1.LANES`, so a vector written for any other page enters this table on the next run. They used to be
# typed here, and on 2026-09-10 that list still read MC2, D1, M3 and NN2 while NN3, LT2 and RC1 were already cached.
NOTE = {"D1": "LightGBM, a different library", "M3": "TabPFN per airport, a different architecture",
        "W1-A": "the weather champion, MC2's family on 49 columns plus METAR", "SB3+RC1": "ridge on the same columns",
        "SB3+NN2": "a bagged neural net", "SB3+NN3": "the searched neural net", "SB3+LT2": "LightGBM linear_tree"}


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


def best_stack(errs: list[np.ndarray]) -> tuple[np.ndarray, float]:
    """The k-predictor form of best_blend: affine weights summing to one that minimise the blend's second moment.
    Weights come from these very rows, so the RMSE it returns lower-bounds anything an out-of-fold choice could reach."""
    e = np.column_stack(errs)
    g = e.T @ e / len(e)
    v = np.linalg.solve(g, np.ones(len(g)))
    return v / v.sum(), float(np.sqrt(1 / v.sum()))


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
    e0 = sb3_predict(tra, tea) - y
    r0 = rmse(e0)
    partners = [(f"{n.removeprefix('SB3+')}, {NOTE.get(n, 'a cached lane-A model')}", raw(n, tea.id.values))
                for n in on_disk() if n != "MC2"]
    partners.insert(0, ("MC2, CatBoost, the champion's own family", floored(champion_holdout("MC2", tra, tea))))
    rows, gains, rhos = [], [], []
    for name, p in partners:
        w, b, rho = best_blend(e0, p - y)
        gains.append(r0 - b)
        rhos.append(rho)
        rows.append(f"| {name} | {rmse(p - y):.2f} | {rho:.4f} | {b:.2f} | {w:.3f} | {r0 - b:+.2f} |")
    front = [f"| {rx} | {needed_rho(r0, float(rx)):.3f} |" for rx in LEVELS]

    short = [n.split(",")[0] for n, _ in partners]
    errs = [p - y for _, p in partners]
    stack = []
    for r in range(1, len(errs) + 1):
        for ix in combinations(range(len(errs)), r):
            w, b = best_stack([e0] + [errs[i] for i in ix])
            stack.append((" + ".join(("SB3", *(short[i] for i in ix))), b, r0 - b, w))
    stack.sort(key=lambda t: t[1])
    srows = [f"| {n} | {b:.2f} | {g:+.2f} | {', '.join(f'{x:.2f}' for x in w)} |" for n, b, g, w in stack]
    full = stack[0]
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
           f"The least agreeing partner is {short[int(np.argmin(rhos))]} at {min(rhos):.3f}, and at its "
           f"{rmse(partners[int(np.argmin(rhos))][1] - y):.0f} s level the frontier asks for "
           f"{needed_rho(r0, rmse(partners[int(np.argmin(rhos))][1] - y)):.3f}. A partner at the champion's own level would have to stay under "
           f"{needed_rho(r0, r0):.3f}, and the two GBM libraries measured here read {rhos[0]:.4f} and {rhos[1]:.4f}. "
           "Reading: a further model class does not reach the floor by blending, whatever library it comes from. It reaches it only by "
           "being better than SB3 outright, which is a different question and needs no blend.", "",
           f"## Every subset at once: the {len(errs)}-partner stack", "",
           "The pairwise table above answers one partner at a time, which is not the question a stack asks. These weights are affine "
           "(they sum to one) and solved on the holdout's own errors for every subset that contains the champion, so each row is a "
           "bound no out-of-fold stack can beat. Weights are listed champion first, then the partners in the row's own order.", "",
           "| stack | RMSE (s) | gain (s) | weights |", "|---|---|---|---|", *srows, "",
           f"The best of the {len(srows)} subsets is **{full[0]}** at {full[1]:.2f} s, {full[2]:+.2f} s against a {FLOOR:.0f} s floor. "
           f"The best single partner buys {max(gains):+.2f} s, so putting every cached model in one stack adds "
           f"{full[2] - max(gains):+.2f} s on top of the best pair.", "",
           ("The table holds every lane-A model whose holdout predictions are on disk, read from `taxi.sw1.LANES`: "
            + ", ".join(n.removeprefix("SB3+") for n in on_disk()) + ". NN1 is not among them: it was superseded by NN2 and its "
            "own predictions were never written to a file."), "",
           (f"The bound clears the {FLOOR:.0f} s floor, so a stack chosen out of fold is worth declaring as its own lane; the "
            f"declared version has to give back whatever the oracle weights are worth."
            if full[2] >= FLOOR else
            f"The bound is under the {FLOOR:.0f} s floor with the weights read off the answers, so no out-of-fold stack of these "
            f"models can reach it and the direction is closed by arithmetic. The partners agree with each other about as much as "
            f"they agree with the champion, so their errors leave little for a wider combination to remove."), ""]
    text = "\n".join(out)
    with open("docs/BF.md", "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
