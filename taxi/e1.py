"""E1: blend of MC2 and M3 (TabPFN per airport); the weight on MC2 is chosen on the month-pair OOF predictions over an 11-point grid, then the holdout
is touched once at k = 11. Named TabPFN-E1 when M3 carries weight (Prior Labs licence, protocol M3 row)."""
from __future__ import annotations

from taxi.champ import champion_holdout, champion_oof, lane_a
from taxi.e2 import blend, pick
from taxi.m2 import is_tail
from taxi.m3 import holdout, oof
from taxi.verdict import paired_delta, rmse
from taxi.w1 import verdict_line

WEIGHTS = tuple(round(0.1 * i, 1) for i in range(11))
K = len(WEIGHTS)


def main() -> None:
    tra, tea = lane_a()
    o3, _ = oof(tra)
    h3, _ = holdout(tra, tea)
    w, table = pick(tra.y.values, champion_oof("MC2", tra), o3, WEIGHTS)
    y = tea.y.values
    h2 = champion_holdout("MC2", tra, tea)
    e0, e1 = h2 - y, blend(h2, h3, w) - y
    v = paired_delta(e0, e1, tea.day.values)
    name = "TabPFN-E1" if w < 1 else "E1"
    vt, jan, jul = is_tail(tea), tea.month.values == 1, tea.month.values == 7
    row = lambda n, e: f"| {n} | {rmse(e):.1f} | {rmse(e[jan]):.1f} | {rmse(e[jul]):.1f} | {rmse(e[vt]):.1f} | {rmse(e[~vt]):.1f} |"  # noqa: E731
    report = ["# E1: blend of MC2 and M3 (TabPFN per airport), Jan+Jul 2025 development holdout", "",
              f"Regenerate with `python -m taxi.e1` (after `python -m taxi.m3`). The weight on MC2 is chosen among {WEIGHTS} on the two models' month-pair "
              f"OOF predictions over the ten training months (MC2: `taxi.champ.champion_oof`; M3: `taxi.m3.oof`); only the chosen weight meets the holdout, "
              f"where MC2 is the cached refit and M3 the per-airport holdout fits (`taxi.m3.holdout`). The choice is priced by k = {K}. "
              "Built with PriorLabs-TabPFN (Apache 2.0 plus attribution).", "",
              "| weight on MC2 | training OOF RMSE (s) |", "|---|---|"] + [f"| {a} | {r:.2f} |" for a, r in table] + ["",
              f"## Holdout verdict vs champion MC2 ({len(tea)} rows; tail rows {int(vt.sum())}; chosen weight {w})", "",
              "| model | RMSE | Jan | Jul | tail rows | body rows |", "|---|---|---|---|---|---|", row("MC2", e0), row("M3", h3 - y), row(f"{name} (w = {w})", e1), "",
              verdict_line(name, "MC2", v, k=K)]
    text = "\n".join(report) + "\n"
    with open("docs/E1.md", "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
