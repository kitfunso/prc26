"""MDE: the smallest improvement each lane's verdict machinery can actually see (diagnostic, never a verdict).

Protocol Stage 0 asks for the detectable-effect floor before any modelling; it was never written down. Lane B holds 49% of the
composite's squared error on 1.6% of the rows, and two of those rows hold half of the lane, so the question is not academic.
"""
from __future__ import annotations

import re
import statistics as st

import numpy as np
import pandas as pd

from taxi.cb1 import lane_b
from taxi.champ import CURRENT, champion_holdout, lane_a, predict_b
from taxi.l2 import floored
from taxi.ceil import PER_A, PER_B
from taxi.m2 import is_tail
from taxi.verdict import paired_delta, promote_floor, rmse

REPS = 2000


def crossing(e0: np.ndarray, days: np.ndarray, mask: np.ndarray, test) -> tuple[float, float]:
    """Smallest fraction the subset's errors must shrink by for `test` to pass, by bisection on a monotone quantity."""
    lo, hi = 0.0, 1.0
    for _ in range(30):
        d = (lo + hi) / 2
        e1 = e0.copy()
        e1[mask] *= 1 - d
        lo, hi = (lo, d) if test(paired_delta(e0, e1, days, reps=REPS)) else (d, hi)
    e1 = e0.copy()
    e1[mask] *= 1 - hi
    return hi, rmse(e0) - rmse(e1)


def block(name: str, e0: np.ndarray, days: np.ndarray, sets: list[tuple[str, np.ndarray]], price) -> list[str]:
    r0, fl = rmse(e0), promote_floor(name)
    out = [f"| rows improved | n | their RMSE now | interval clears zero at | {fl:.0f} s floor met at | binding | their RMSE then | composite (s) |",
           "|---|---|---|---|---|---|---|---|"]
    for label, mask in sets:
        sub = rmse(e0[mask])
        d_ci, g_ci = crossing(e0, days, mask, lambda v: v.lo > 0)
        d_fl, g_fl = crossing(e0, days, mask, lambda v: v.delta >= fl)
        d, g = (d_ci, g_ci) if d_ci > d_fl else (d_fl, g_fl)
        bind = "interval" if d_ci > d_fl else f"{fl:.0f} s floor"
        out.append(f"| {label} | {int(mask.sum()):,} | {sub:,.1f} s | {d_ci * 100:.2f}% ({g_ci:+.2f} s) | "
                   f"{d_fl * 100:.2f}% ({g_fl:+.2f} s) | {bind} | {sub * (1 - d):,.1f} s | {price(g):+.3f} |")
    return [f"### Lane {name}, champion {CURRENT[name]} at {r0:,.2f} s", ""] + out + [""]


def ledger_se(path: str = "docs/TRIAL-LEDGER.md") -> dict[str, list[float]]:
    """Every holdout verdict's recorded paired SE, by lane: the noise a real challenger of this campaign actually faced."""
    out: dict[str, list[float]] = {"A": [], "B": []}
    for ln in open(path, encoding="utf-8"):
        c = [x.strip() for x in ln.split("|")]
        m = re.search(r"paired SE ([\d.]+) s", ln)
        if ln.startswith("| H") and m and len(c) > 3 and c[3] in out:
            out[c[3]].append(float(m.group(1)))
    return out


def main(out: str = "docs/MDE.md") -> None:
    fl_a, fl_b = promote_floor("A"), promote_floor("B")
    tra, tea = lane_a()
    trb, teb = lane_b()
    ea = floored(champion_holdout(CURRENT["A"], tra, tea)) - tea.y.values
    eb = floored(predict_b(CURRENT["B"], trb, teb)) - teb.y.values
    na, nb = len(ea), len(eb)
    sa, sb = float((ea ** 2).sum()), float((eb ** 2).sum())
    n, comp = na + nb, float(np.sqrt((sa + sb) / (na + nb)))
    price = lambda lane, g: comp - float(np.sqrt((((rmse(ea) - g) ** 2 if lane == "A" else rmse(ea) ** 2) * na  # noqa: E731
                                                 + ((rmse(eb) - g) ** 2 if lane == "B" else rmse(eb) ** 2) * nb) / n))

    ta = is_tail(tea)
    sets_a = [("every row", np.ones(na, bool)), ("body rows only (|c| under 600 s)", ~ta), ("tail rows only", ta)]
    ord_b = teb.y.values < 3600
    sets_b = [("every row", np.ones(nb, bool)), ("ordinary rows (truth under 1 h)", ord_b),
              ("the rest (truth over 1 h)", ~ord_b)]

    worst = np.argsort(np.abs(eb))[::-1]
    share = [f"| worst {k} rows | {float((eb[worst[:k]] ** 2).sum()) / sb * 100:.1f}% |" for k in (1, 2, 5, 20, 100)]
    se = ledger_se()
    med = {k: st.median(v) for k, v in se.items()}
    fl = {k: promote_floor(k) for k in se}
    real = [f"| {k} | {len(v)} | {min(v) * 1.96:.1f} s | {med[k] * 1.96:.1f} s | {max(v) * 1.96:.1f} s | {fl[k]:.0f} s | "
            f"{format(fl[k], '.0f') + ' s floor' if med[k] * 1.96 < fl[k] else 'the interval, by ' + format(med[k] * 1.96 / fl[k], '.1f') + 'x'} |"
            for k, v in se.items()]

    lines = ["# MDE: the smallest gain each lane's verdict can see (diagnostic, never a verdict)", "",
             f"Regenerate with `python -m taxi.mde`. Champions {CURRENT['A']} and {CURRENT['B']}, floored, holdout Jan + Jul 2025. "
             f"No cell is declared and nothing here decides a verdict. Protocol Stage 0 asks for the detectable-effect floor before "
             f"any modelling and this campaign never wrote one down. The decision rule needs BOTH halves: the 95% paired day-block "
             f"interval above zero AND a point gain of at least the lane's own floor, {fl_a:.0f} s on lane A and {fl_b:.0f} s on "
             f"lane B, which are the same {2.0 * PER_A:.2f} board seconds priced at each lane's measured carry. This prices what a challenger must "
             f"actually deliver to satisfy them, by shrinking the champion's own errors on a named set of rows until the rule "
             f"fires. Shrinking real errors is the most generous challenger there is: it never mis-fires on a row it does not "
             f"improve, so every number below is a LOWER bound on what a real model has to do.", "",
             "## 1. What the rule demands", "",
             *block("A", ea, tea.day.values, sets_a, lambda g: price("A", g)),
             *block("B", eb, teb.day.values, sets_b, lambda g: price("B", g)), "",
             f"Every interval clears zero at the first fraction the bisection tries, so section 1 measures the FLOOR half of the rule "
             f"and nothing else. A proportional shrink improves every row at once, so the paired difference is positive on every day "
             f"and the bootstrap has no disagreement to price. Real challengers move rows both ways, and section 2 reads their noise "
             f"off the campaign's own verdicts instead of inventing it.", "",
             "## 2. What the interval actually cost, on this campaign's real challengers", "",
             f"Every holdout verdict in `docs/TRIAL-LEDGER.md` that recorded a paired standard error, grouped by lane. The "
             f"minimum detectable effect is 1.96 SE: below it a challenger's interval contains zero however good its point estimate is.", "",
             "| lane | verdicts | smallest MDE | median MDE | largest MDE | promotion floor | what binds |", "|---|---|---|---|---|---|---|", *real, "",
             f"Lane A's machinery is not the constraint: its median challenger had to beat {med['A'] * 1.96:.1f} s of lane RMSE to clear "
             f"zero, and the rule asks for {fl_a:.0f}. On lane B the two halves are now close: its median challenger had to beat "
             f"{med['B'] * 1.96:.0f} s against a floor of {fl_b:.0f} s, so {'the interval' if med['B'] * 1.96 > fl_b else 'the floor'} binds. "
             f"Under the flat 2 s floor this was never a contest, because 2 s of lane B is {2.0 * PER_B:.2f} board seconds and the "
             f"interval asked for {med['B'] * 1.96 * PER_B:.2f}.", "",
             "## 3. Why lane B's interval is so wide", "",
             "| rows | share of lane-B squared error |", "|---|---|", *share, "",
             f"Lane B is {nb:,} rows against lane A's {na:,}, and its squared error is concentrated in a handful of them "
             f"(`docs/DE.md`, flag F21). A day-block bootstrap resamples whole days, so a day carrying one of those rows moves the "
             f"whole lane's RMSE by hundreds of seconds, and the interval inherits that spread whatever the challenger does "
             f"elsewhere.", ""]
    text = "\n".join(lines) + "\n"
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
