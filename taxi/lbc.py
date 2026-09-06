"""LBC: should the copy mixture reach the nine airports L4's gate never touches?

Diagnostic. No cell declared, no champion moves, no verdict decided. `taxi.l4.touched` admits only no-type rows at airports with
at least MIN_NOTYP of them, which today is LIRF alone, so L6a and L8c are identical to L4b at the other nine. LFPG carries most of
the lane's squared error and has never been offered the branch. Regenerate with `python -m taxi.lbc`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from taxi.cb1 import lane_b
from taxi.champ import CURRENT, champion_holdout, lane_a, predict_b
from taxi.l2 import floored
from taxi.l3 import COPY
from taxi.l4 import MIN_NOTYP, touched


def branch_err(e0: np.ndarray, y: np.ndarray, z: np.ndarray, m: np.ndarray) -> np.ndarray:
    """The copy branch predicts z_sched, so a near copy still carries its residual; only an exact copy makes the error zero."""
    e = e0.copy()
    e[m] = z[m] - y[m]
    return e


def known_err(e0: np.ndarray, m: np.ndarray) -> np.ndarray:
    """The strictly looser ceiling: membership AND the label. Reported beside the branch so the gap between them is visible."""
    e = e0.copy()
    e[m] = 0.0
    return e


def census(trb: pd.DataFrame, teb: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    y, z = teb.y.values, teb.z_sched.values
    sse = (pred - y) ** 2
    m = touched(trb, teb)
    d = pd.DataFrame({"apt": teb.apt.values, "sse": sse, "exact": np.isclose(y, z), "near": np.abs(y - z) <= COPY,
                      "touched": m, "notyp": teb.notyp.values == 1})
    g = d.groupby("apt").agg(rows=("sse", "size"), sse=("sse", "sum"), exact=("exact", "sum"), near=("near", "sum"),
                             touched=("touched", "sum"), notyp=("notyp", "sum"))
    g["fit_notyp"] = trb[trb.notyp == 1].groupby("apt").size().reindex(g.index).fillna(0).astype(int)
    g["share"] = 100 * g.sse / g.sse.sum()
    g["rmse"] = np.sqrt(g.sse / g.rows)
    return g.sort_values("sse", ascending=False)


def main() -> None:
    tra, tea = lane_a()
    trb, teb = lane_b()
    pred = floored(predict_b(CURRENT["B"], trb, teb))
    ea = floored(champion_holdout(CURRENT["A"], tra, tea)) - tea.y.values
    y, z = teb.y.values, teb.z_sched.values
    na, nb = len(ea), len(y)
    sa = float((ea ** 2).sum())

    def comp(err_b: np.ndarray) -> float:
        return float(np.sqrt((sa + float((err_b ** 2).sum())) / (na + nb)))

    g = census(trb, teb, pred)
    e0 = pred - y
    branch = lambda m: branch_err(e0, y, z, m)
    known = lambda m: known_err(e0, m)
    base_b, base_c = float(np.sqrt(np.mean(e0 ** 2))), comp(e0)

    rows = []
    for name, m in (("exact copies, y = z_sched", np.isclose(y, z)), (f"near copies, |y - z_sched| <= {COPY:.0f} s", np.abs(y - z) <= COPY)):
        eb, ek = branch(m), known(m)
        rows.append(f"| {name} | {int(m.sum()):,} | {100 * m.mean():.3f}% | {100 * (e0[m] ** 2).sum() / (e0 ** 2).sum():.2f}% | "
                    f"{base_b - float(np.sqrt(np.mean(eb ** 2))):+.1f} | {base_c - comp(eb):+.3f} | "
                    f"{base_b - float(np.sqrt(np.mean(ek ** 2))):+.1f} | {base_c - comp(ek):+.3f} |")

    # The decision is whether to EXTEND the branch, so every oracle is split by the gate rather than reported over the lane.
    tch = touched(trb, teb)
    near = np.abs(y - z) <= COPY
    splits = []
    for nm, m in (("exact copies", np.isclose(y, z)), ("near copies", near)):
        for where, sel in (("already touched (LIRF)", m & tch), ("the nine untouched airports", m & ~tch)):
            e = branch(sel)
            splits.append(f"| {nm}, {where} | {int(sel.sum()):,} | {100 * (e0[sel] ** 2).sum() / (e0 ** 2).sum():.2f}% | "
                          f"{base_b - float(np.sqrt(np.mean(e ** 2))):+.1f} | {base_c - comp(e):+.3f} |")

    out = [
        "# LBC: the copy mixture at the nine airports the gate never touches (diagnostic, never a verdict)",
        "",
        "Regenerate with `python -m taxi.lbc`. No cell is declared here and no champion moves.",
        f"Champion {CURRENT['B']} floored, {nb:,} lane-B holdout rows, RMSE {base_b:.4f} s, composite {base_c:.4f} s.",
        "",
        "## 1. Where the error is, and where the branch is allowed to run",
        "",
        f"`taxi.l4.touched` admits a row only if it has no aircraft type AND its airport has at least {MIN_NOTYP} such rows in training.",
        "That is a gate on data availability, not on where the copies are, so it is worth checking what it excludes.",
        "",
        g.to_string(float_format=lambda v: f"{v:9.2f}"),
        "",
        "## 2. The oracle: perfect membership and the exact value on every copy row",
        "",
        "| copy definition | rows | share of lane | share of lane SSE | branch: predict z_sched, lane B | composite | ceiling: know y, lane B | composite |",
        "|---|---|---|---|---|---|---|---|",
        *rows,
        "",
        "## 3. The decision: what EXTENDING the branch to the other nine airports would buy",
        "",
        "| oracle | rows | share of lane SSE | lane-B gain (s) | composite gain (s) |",
        "|---|---|---|---|---|",
        *splits,
        "",
        "## 4. Reading",
        "",
        "The gate is not a statement about where copies are. `touched` needs a row with no aircraft type at an airport carrying "
        f"{MIN_NOTYP} such rows in training, and only LIRF clears it ({int(g.loc['LIRF', 'fit_notyp'])} no-type fit rows against 0 to 6 everywhere else). "
        "No-type is a data-availability marker, and the copy behaviour plainly is not confined to it: LTFM has "
        f"{int(g.loc['LTFM', 'near'])} near copies in {int(g.loc['LTFM', 'rows'])} holdout rows with {int(g.loc['LTFM', 'fit_notyp'])} no-type fit rows, "
        f"EHAM {int(g.loc['EHAM', 'near'])} in {int(g.loc['EHAM', 'rows'])}, LSZH {int(g.loc['LSZH', 'near'])} in {int(g.loc['LSZH', 'rows'])}.",
        "",
        "Two answers, and they point opposite ways.",
        "",
        "The EXACT-copy extension is dead. LFPG is the obvious target because it carries "
        f"{g.loc['LFPG', 'share']:.1f}% of the lane's squared error, and it holds {int(g.loc['LFPG', 'exact'])} exact copy in {int(g.loc['LFPG', 'rows'])} rows. "
        "Across all nine untouched airports there are 5 of them, worth +0.006 s of composite with the answer in hand. Where the "
        "error is, the copies are not; where the copies are, the error is not. This closes the reflex that F28 and `docs/L8.md` "
        "invite, that the LIRF-only mixture should simply be given to the other airports.",
        "",
        "The NEAR-copy extension is not dead and should not be written off with it. At the nine untouched airports 746 rows sit "
        f"within {COPY:.0f} s of `z_sched`, holding 3.18% of the lane's squared error, and predicting `z_sched` on exactly those rows is worth "
        "+25.4 s of lane B and +2.211 s of composite. That is a perfect-membership ceiling and a real classifier collects a "
        "fraction of it, but the ceiling is above the 2 s floor's composite equivalent for this lane (0.175 s, `docs/LEV.md`) by "
        "more than an order of magnitude, and those nine airports have no copy handling of any kind today. The honest conclusion "
        "is a declared lane, not a closure.",
        "",
    ]
    text = "\n".join(out) + "\n"
    with open("docs/LBC.md", "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
