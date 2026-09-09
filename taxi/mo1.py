"""MO1: monotone constraints on lane A, the register's isotonic/monotone row, closed by measurement instead of by argument.

`docs/MONO.md` measured that the physics sign REVERSES between the level `y` and the target `c`, and then left the rung open on
two conditions: the sign has to come from rows the verdict does not judge, and the constraint is regularisation that binds when
data is thin. This module meets the first condition and tests the second. Two arms, both k = 1: M1's recorded lane-A cell
unconstrained, and the same cell with a monotone constraint on each domain-knowledge column whose sign that fold's own training
rows carry. The unconstrained arm is the control, so the constraint's effect is the difference between them and not a claim.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from taxi.champ import COLS_D2, lane_a
from taxi.fam1 import BUDGET_S, FOLDS, GATE, oof, score_arm
from taxi.l2 import floored
from taxi.m1 import chosen_cells, matrix, train
from taxi.mono import CAND, FLIP
from taxi.nn1 import base_oof
from taxi.sb3 import predict as sb3_predict
from taxi.verdict import paired_delta, rmse
from taxi.w1 import verdict_line

K = 1
METHOD = "advanced"  # protocol rung 6 names intermediate or advanced; advanced constrains least and so tests the rung hardest
SEEN: list[dict] = []


def constraints(tr: pd.DataFrame) -> list[int]:
    """One sign per column of `COLS_D2`, read off THIS fold's training rows only, so no judged row chose its own constraint."""
    got = {}
    for col, _want in CAND:
        if col not in tr.columns:
            continue
        v = tr[col].values.astype("float64")
        ok = np.isfinite(v)
        rho = float(spearmanr(v[ok], tr.c.values[ok]).statistic)
        got[col] = int(np.sign(rho)) if abs(rho) >= FLIP else 0
    SEEN.append(dict(n=len(tr), **got))
    return [got.get(c, 0) for c in COLS_D2]


def _fit(tr: pd.DataFrame, te: pd.DataFrame, mono: list[int] | None) -> np.ndarray:
    cell, rounds = chosen_cells()["A"]
    if mono is not None:
        cell = {**cell, "monotone_constraints": mono, "monotone_constraints_method": METHOD}
    bst = train(tr, "A", cell, rounds=rounds, cols=COLS_D2)
    return bst.predict(matrix(te, COLS_D2))


def arm_free(tr: pd.DataFrame, te: pd.DataFrame) -> np.ndarray:
    return _fit(tr, te, None)


def arm_mono(tr: pd.DataFrame, te: pd.DataFrame) -> np.ndarray:
    return _fit(tr, te, constraints(tr))


def main(out: str = "docs/MO1.md") -> None:
    tra, tea = lane_a()
    base, t0 = base_oof(tra), time.time()
    print("| arm | own RMSE | blend | MC2 base | agreement | min |\n|---|---|---|---|---|---|", flush=True)

    rows = {}
    for tag, label, fn in (("free", "M1 lane-A cell, unconstrained (the control)", arm_free),
                           ("mono", "the same cell, monotone on the measured sign", arm_mono)):
        p, spent, note = oof(tra, fn, BUDGET_S / 2)
        pd.DataFrame({"id": tra.id.values, "pred": p}).to_parquet(f"data/cache/mo1_{tag}_oof.parquet", index=False)
        rows[tag] = score_arm(tra, base, tag, label, p, spent, note)

    gain = rows["mono"]["base"] - rows["mono"]["blend"]
    cost = rows["mono"]["level"] - rows["free"]["level"]

    verd = []
    if gain >= GATE:
        y, sb3, w = tea.y.values, sb3_predict(tra, tea), rows["mono"]["w"]
        lvl = floored(arm_mono(tra, tea) + tea.z_aobt.values)
        v = paired_delta(sb3 - y, floored(w * lvl + (1 - w) * sb3) - y, tea.day.values)
        verd = [f"- **MO1-mono + SB3** at w={w:.4f}, holdout {rmse(lvl - y):.2f} s alone: "
                + verdict_line("MO1-mono", "SB3", v, k=K)]

    sign_tbl = ["| fold | rows | " + " | ".join(c for c, _ in CAND) + " |",
                "|---|---|" + "---|" * len(CAND)]
    for fold, s in zip(FOLDS, SEEN):
        sign_tbl.append(f"| {fold} | {s['n']:,} | "
                        + " | ".join(f"{s.get(c, 0):+d}" if s.get(c, 0) else "0" for c, _ in CAND) + " |")

    text = "\n".join([
        "# MO1: monotone constraints on lane A, the register's last EMPTY row", "",
        f"Regenerate with `python -m taxi.mo1`. Declared in protocol section 4 before the code was written. Two arms, both a "
        f"FIXED cell with no search, k = {K}: M1's recorded lane-A cell {chosen_cells()['A'][0]} at "
        f"{chosen_cells()['A'][1]} rounds, once free and once with `monotone_constraints_method=\"{METHOD}\"`. Folds are the "
        f"three declared month pairs {FOLDS}; the comparator is MC2's cached out-of-fold vector on those rows, the same one "
        f"FAM1, FAM2 and FAM3 used. Total {(time.time() - t0) / 60:.0f} min.", "",
        f"`docs/MONO.md` left this rung open on two conditions. The first is met here by construction: the sign is a Spearman "
        f"rank correlation against `c` on each fold's OWN training rows, so no judged row ever chose the constraint imposed on "
        f"it, and a column whose |rho| falls under {FLIP} is left free. The second condition is the question this file answers, "
        f"because a constraint is regularisation and lane A fits {len(tra):,} rows.", "",
        "## The sign each fold read, and imposed", "", *sign_tbl, "",
        "## The screen", "",
        f"Gate declared in advance and identical to FAM1, FAM2 and FAM3: an arm meets the untouched holdout only if its blend "
        f"with MC2 beats MC2 alone by {GATE:.1f} s on these rows. A failure is a closed register row and costs no holdout fit.", "",
        "| arm | what it is | own RMSE | blend with MC2 | gain over MC2 | unclipped weight | agreement | outcome |",
        "|---|---|---|---|---|---|---|---|",
        *[f"| MO1-{t} | {r['label']} | {r['level']:.2f} | {r['blend']:.2f} | {r['base'] - r['blend']:+.2f} | "
          f"{r['raw']:+.4f} | {r['agree']:.4f} | "
          f"{'PASS' if r['base'] - r['blend'] >= GATE else 'closed at this cell'}"
          f"{'; ' + r['note'] if r['note'] else ''} |" for t, r in rows.items()], "",
        "## What the constraint itself cost", "",
        f"The two arms differ in one parameter, so the constraint's own effect is the difference between them and not a "
        f"comparison across families: **{cost:+.2f} s** of out-of-fold lane A, {rows['free']['level']:.2f} free against "
        f"{rows['mono']['level']:.2f} constrained.", "",
        ("## Verdict against SB3" if verd else "## The arm failed the declared gate"), "",
        *(verd or [f"The constrained blend with MC2 buys {gain:+.2f} s against a {GATE:.1f} s gate, so the arm never met the "
                   f"holdout. The register row closes on a measured constrained fit rather than on the argument that a "
                   f"constraint would be imposed backwards."]), ""])
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
