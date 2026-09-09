"""SH1 (protocol section 4, diagnostic): the selection-honest re-judgement Stage 5.3 asks for and this campaign never ran.

58 holdout verdicts have read the same 339,046 lane-A rows. Stage 5.3 says choose on early rows, judge on untouched late
rows, and report the haircut as its own number. The holdout is two whole months, so the split costs no fit: nominate on one
month, judge on the other, and compare against the champion the campaign promoted while reading both.
Nothing is chosen here and nothing can be promoted or demoted by it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from taxi.cb1 import lane_b
from taxi.champ import CURRENT, lane_a, predict_b
from taxi.l2 import floored
from taxi.sb3 import predict as sb3_predict
from taxi.sw1 import family
from taxi.verdict import rmse

DIRECTIONS = ((1, 7), (7, 1))
NAMES = {1: "January", 7: "July"}


def lane_b_family(trb: pd.DataFrame, teb: pd.DataFrame) -> dict[str, np.ndarray]:
    """Every lane-B cell the champion registry can still reach, refit on the training months exactly as its verdict did."""
    from taxi import l4, l6, l8
    names = ["B4", *l4.CELLS, *l6.CELLS, *l8.CELLS]
    return {n: floored(predict_b(n, trb, teb)) for n in dict.fromkeys(names)}


def split_table(fam: dict[str, np.ndarray], y: np.ndarray, month: np.ndarray) -> list[str]:
    rows = []
    for n, p in fam.items():
        per = [rmse(p[month == m] - y[month == m]) for m in (1, 7)]
        rows.append(f"| {n} | {per[0]:,.2f} | {per[1]:,.2f} | {rmse(p - y):,.2f} |")
    return rows


def one_direction(fam: dict[str, np.ndarray], y: np.ndarray, month: np.ndarray, champ: str,
                  nom: int, jud: int) -> tuple[str, str]:
    """Pick on the nominate month alone, then report that pick on rows the pick never saw."""
    a, b = month == nom, month == jud
    pick = min(fam, key=lambda n: rmse(fam[n][a] - y[a]))
    honest, as_run = rmse(fam[pick][b] - y[b]), rmse(fam[champ][b] - y[b])
    line = (f"| nominate on {NAMES[nom]}, judge on {NAMES[jud]} | {pick} | {honest:,.2f} | {champ} | {as_run:,.2f} | "
            f"{honest - as_run:+,.2f} |")
    note = (f"{NAMES[nom]} nominates **{pick}**" +
            ("" if pick == champ else f", NOT the promoted {champ}") +
            f"; on {NAMES[jud]} that costs {honest - as_run:+,.2f} s against the promoted champion.")
    return line, note


def lane_section(tag: str, fam: dict[str, np.ndarray], y: np.ndarray, month: np.ndarray, champ: str) -> list[str]:
    lines, notes = zip(*(one_direction(fam, y, month, champ, n, j) for n, j in DIRECTIONS))
    return [f"## Lane {tag}: {len(fam)} models, {len(y):,} holdout rows, promoted champion {champ}", "",
            "| model | January RMSE | July RMSE | both months |", "|---|---|---|---|",
            *split_table(fam, y, month), "",
            "| direction | nominated on one month alone | its RMSE on the judged month | promoted while reading both | "
            "its RMSE on the same rows | haircut |", "|---|---|---|---|---|---|", *lines, "",
            *[f"- {n}" for n in notes], ""]


def main(dst: str = "docs/SH1.md") -> None:
    tra, tea = lane_a()
    fam_a = {"SB3": floored(sb3_predict(tra, tea))}
    fam_a.update(family(tea, sb3_predict(tra, tea)))
    trb, teb = lane_b()

    out = ["# SH1: the selection-honest re-judgement, Jan + Jul 2025 development holdout", "",
           "Diagnostic, never a verdict, no cell chosen. Regenerate with `python -m taxi.sh1`. Declared in protocol section 4 "
           "before the code was written. Stage 5.3 of the protocol asks that a spec be nominated on early rows and judged on "
           "untouched late rows, with the haircut reported as its own number, and this campaign had never run it: the whole "
           "repo returned zero matches for \"selection-honest\". The development holdout is two whole months, so the split "
           "needs no new fit. Nomination reads ONE month and picks the `argmin` of its RMSE; the judged number is that "
           "model's RMSE on the other month, which the nomination never saw. The as-run column is the champion this campaign "
           "actually promoted while reading both months, scored on the same judged rows. A positive haircut means the "
           "campaign's reported numbers are optimistic by that much.", "",
           *lane_section("A", fam_a, tea.y.values, tea.month.values, CURRENT["A"]),
           *lane_section("B", lane_b_family(trb, teb), teb.y.values, teb.month.values, CURRENT["B"])]
    text = "\n".join(out)
    with open(dst, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
