"""L11: the copy mixture reaches the nine airports L4's gate excludes.

`taxi.l4.touched` needs a no-type row at an airport with at least MIN_NOTYP of them in training, so LIRF alone clears it and L6a and
L8c are identical to L4b everywhere else. LBC section 3 prices the extension at +25.4 s of lane B and +2.211 s of composite with
perfect membership. One change, no search: LIRF keeps exactly what L8c gives it, the other nine become touchable on all their fit
rows. Declared in protocol section 4 before any fit; regenerate with `python -m taxi.l11`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from taxi.cb1 import lane_b
from taxi.champ import CURRENT, predict_b
from taxi.data import CV_FOLDS
from taxi.l2 import floored
from taxi.l4 import touched
from taxi.l6 import CELLS as L6_CELLS, per_row
from taxi.l8 import gbm
from taxi.verdict import paired_delta, rmse
from taxi.w1 import verdict_line


def served(tr: pd.DataFrame, te: pd.DataFrame) -> list:
    """The airports the gate already serves. They are left alone so the cell cannot regress the only branch in production."""
    return np.unique(te.apt.values[touched(tr, te)]).tolist()


def predict(tr: pd.DataFrame, te: pd.DataFrame) -> np.ndarray:
    q = served(tr, te)

    def mask(t: pd.DataFrame, e: pd.DataFrame) -> np.ndarray:
        return touched(t, e) | ~np.isin(e.apt.values, q)

    def fit_rows(t: pd.DataFrame, apt: object) -> pd.DataFrame:
        sel = (t.apt == apt) & (t.notyp == 1) if apt in q else t.apt == apt
        return t[sel].reset_index(drop=True)

    return per_row(tr, te, L6_CELLS["L6a"], prob=gbm, mask=mask, fit_rows=fit_rows)


def by_airport(te: pd.DataFrame, base: np.ndarray, l11: np.ndarray) -> pd.DataFrame:
    y = te.y.values
    d = pd.DataFrame({"apt": te.apt.values, "b": (base - y) ** 2, "l": (l11 - y) ** 2, "moved": ~np.isclose(base, l11)})
    g = d.groupby("apt").agg(rows=("b", "size"), moved=("moved", "sum"), b=("b", "sum"), l=("l", "sum"))
    g["champ_rmse"], g["l11_rmse"] = np.sqrt(g.b / g.rows), np.sqrt(g.l / g.rows)
    g["sse_share"] = 100 * g.b / g.b.sum()
    return g.drop(columns=["b", "l"]).sort_values("sse_share", ascending=False)


def main() -> None:
    trb, teb = lane_b()
    q = served(trb, teb)
    print(f"lane B: {len(trb):,} training rows, {len(teb):,} holdout rows; gate already serves {q}", flush=True)

    lines = [f"| fold | rows | {CURRENT['B']} RMSE (s) | L11 RMSE (s) | rows moved |", "|---|---|---|---|---|"]
    for fold in CV_FOLDS:
        m = trb.month.isin(fold).values
        f, va = trb[~m].reset_index(drop=True), trb[m].reset_index(drop=True)
        y = va.y.values
        b, p = floored(predict_b(CURRENT["B"], f, va)), floored(predict(f, va))
        lines.append(f"| {fold} | {len(va):,} | {rmse(b - y):.1f} | {rmse(p - y):.1f} | {int((~np.isclose(b, p)).sum()):,} |")
        print(lines[-1], flush=True)

    y = teb.y.values
    base, l11 = floored(predict_b(CURRENT["B"], trb, teb)), floored(predict(trb, teb))
    keep = np.isin(teb.apt.values, q)
    same = bool(np.allclose(base[keep], l11[keep]))
    vd = paired_delta(base - y, l11 - y, teb.day.values)
    g = by_airport(teb, base, l11)
    jan, jul = teb.month.values == 1, teb.month.values == 7
    row = lambda n, e: f"| {n} | {rmse(e):.1f} | {rmse(e[jan]):.1f} | {rmse(e[jul]):.1f} |"  # noqa: E731

    out = ["# L11: the copy mixture at the nine airports the gate excludes, Jan + Jul 2025 development holdout", "",
           f"Regenerate with `python -m taxi.l11`. L6a's soft mixture and L8c's CatBoost copy probability throughout, both inherited "
           f"unchanged; the only change is which rows the branch reaches and which training rows fit it. Airports already served by "
           f"`taxi.l4.touched` ({', '.join(map(str, q))}) keep exactly what {CURRENT['B']} gives them, and the check that they do is "
           f"below. k = 1, nothing is chosen here.", "",
           f"**LIRF unchanged: {same}.** {int(keep.sum()):,} holdout rows at the served airports, maximum absolute difference "
           f"{float(np.max(np.abs(base[keep] - l11[keep]))):.6f} s.", "",
           "## Month-pair folds (reported, not used to select)", ""] + lines
    out += ["", f"## Holdout verdict vs champion {CURRENT['B']} ({len(teb):,} rows)", "",
            "| model | RMSE | Jan | Jul |", "|---|---|---|---|", row(CURRENT["B"], base - y), row("L11", l11 - y), "",
            verdict_line("L11", CURRENT["B"], vd, k=1), "", "## By airport", "",
            g.to_string(float_format=lambda v: f"{v:9.2f}"), ""]
    text = "\n".join(out) + "\n"
    with open("docs/L11.md", "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
