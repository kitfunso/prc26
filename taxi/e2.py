"""E2: fixed-weight blend of MC2 and D1; the weight is chosen on the cached month-pair OOF predictions, then the holdout is touched once at k = 3."""
from __future__ import annotations

import numpy as np

from taxi.champ import champion_holdout, champion_oof, lane_a
from taxi.m2 import is_tail
from taxi.verdict import paired_delta, rmse
from taxi.w1 import verdict_line

WEIGHTS = (0.5, 0.7, 0.9)  # weight on MC2, declared before the run (protocol section 4, E2)
K = len(WEIGHTS)


def blend(a: np.ndarray, b: np.ndarray, w: float) -> np.ndarray:
    return w * a + (1 - w) * b


def pick(y: np.ndarray, a: np.ndarray, b: np.ndarray, weights: tuple[float, ...] = WEIGHTS) -> tuple[float, list[tuple[float, float]]]:
    """RMSE per weight on the rows given (the training OOF rows) and the best weight; the choice is priced by the k = len(weights) bound."""
    table = [(w, rmse(blend(a, b, w) - y)) for w in weights]
    return min(table, key=lambda t: t[1])[0], table


def main() -> None:
    tra, tea = lane_a()
    w, table = pick(tra.y.values, champion_oof("MC2", tra), champion_oof("D1", tra))
    y = tea.y.values
    p_mc2, p_d1 = champion_holdout("MC2", tra, tea), champion_holdout("D1", tra, tea)
    e0, e1 = p_mc2 - y, blend(p_mc2, p_d1, w) - y
    v = paired_delta(e0, e1, tea.day.values)
    vt, jan, jul = is_tail(tea), tea.month.values == 1, tea.month.values == 7
    row = lambda name, e: f"| {name} | {rmse(e):.1f} | {rmse(e[jan]):.1f} | {rmse(e[jul]):.1f} | {rmse(e[vt]):.1f} | {rmse(e[~vt]):.1f} |"  # noqa: E731
    report = ["# E2: fixed-weight blend of MC2 and D1, Jan+Jul 2025 development holdout", "",
              f"Regenerate with `python -m taxi.e2`. The weight on MC2 is chosen among {WEIGHTS} on the two champions' cached month-pair OOF predictions "
              f"over the ten training months (`taxi.champ.champion_oof`); only the chosen weight meets the holdout, where both components are the cached "
              f"refits at their recorded rounds. The choice is priced by k = {K}.", "",
              "| weight on MC2 | training OOF RMSE (s) |", "|---|---|"] + [f"| {a} | {r:.2f} |" for a, r in table] + ["",
              f"## Holdout verdict vs champion MC2 ({len(tea)} rows; tail rows {int(vt.sum())}; chosen weight {w})", "",
              "| model | RMSE | Jan | Jul | tail rows | body rows |", "|---|---|---|---|---|---|", row("MC2", e0), row("D1", p_d1 - y), row(f"E2 (w = {w})", e1), "",
              verdict_line("E2", "MC2", v, k=K)]
    text = "\n".join(report) + "\n"
    with open("docs/E2.md", "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
