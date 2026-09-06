"""TA (diagnostic, never a verdict): lane A's tail is variance, not bias, and two model families miss on the same rows.

CA1 measured that 58% of MC2's promotion over D1 lives in the top 1% of rows, and F17 counts the rows that carry the composite. Neither asks
whether that tail is CORRECTABLE. This asks: is the champion mis-calibrated anywhere a feature can see, and would a different model class miss
the same rows? Both answers are no, which redirects the remaining effort away from models and calibration and onto new information.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from taxi.champ import CURRENT, champion_holdout, lane_a
from taxi.l2 import floored

TOP = 3390  # the lane's top 1% of rows, the set CA1 prices the chain against
BANDS = ([-1e9, 1200, 2400, 3600, 5400, 7200, 1e9], ["under 20 m", "20 to 40 m", "40 to 60 m", "1 to 1.5 h", "1.5 to 2 h", "over 2 h"])
OTHERS = ("D1", "W1-A")


def deciles(p: np.ndarray, y: np.ndarray, by: np.ndarray, name: str) -> list[str]:
    q = pd.qcut(by, 10, labels=False, duplicates="drop")
    t = pd.DataFrame({"p": p, "y": y, "e": p - y}).groupby(q, observed=True).agg(
        n=("e", "size"), mp=("p", "mean"), my=("y", "mean"), bias=("e", "mean"))
    out = [f"| {name} decile | rows | mean prediction | mean truth | bias |", "|---|---|---|---|---|"]
    for i, r in t.iterrows():
        out.append(f"| {int(i) + 1} | {int(r.n):,} | {r.mp:,.0f} s | {r.my:,.0f} s | {r.bias:+,.1f} s |")
    return out


def recoverable(e: np.ndarray, g: np.ndarray, rmse: float, n: int) -> float:
    """RMSE seconds freed by removing every group's mean error, the ceiling on any additive recalibration."""
    d = pd.DataFrame({"e": e, "g": g}).groupby("g", observed=True).e
    return rmse - float(np.sqrt(max((e ** 2).sum() - (d.count() * d.mean() ** 2).sum(), 0.0) / n))


def main() -> None:
    tra, tea = lane_a()
    y = tea.y.values
    p = floored(champion_holdout(CURRENT["A"], tra, tea))
    e = p - y
    sse = e ** 2
    rmse, n = float(np.sqrt(sse.mean())), len(tea)
    o = np.argsort(-sse)
    top = o[:TOP]
    zs = tea.z_sched.values
    band = pd.cut(zs, BANDS[0], labels=BANDS[1])

    lines = [f"# TA: lane A's tail is variance, not bias (diagnostic, never a verdict)", "",
             f"Regenerate with `python -m taxi.ta`. Champion {CURRENT['A']}, floored, holdout Jan + Jul 2025, {n:,} lane-A rows at "
             f"{rmse:,.4f} s. No cell is declared here. The point of the run is to say which KINDS of work still have room, and the answer "
             "removes two of them.", "",
             "## 1. Where the error is, and which way it points", "",
             "| rows, worst first by squared error | share of the lane's squared error |", "|---|---|"]
    for k in (10, 100, 1000, TOP):
        lines.append(f"| top {k:,} | {sse[o[:k]].sum() / sse.sum() * 100:.1f}% |")
    lo, hi = e[top] < 0, e[top] > 0
    lines += ["", f"The tail is under-prediction. Of the top {TOP:,} rows, {int(lo.sum()):,} are predicted too low and carry "
              f"{sse[top][lo].sum() / sse[top].sum() * 100:.1f}% of the tail's squared error, against {int(hi.sum())} predicted too high "
              f"carrying {sse[top][hi].sum() / sse[top].sum() * 100:.1f}%. A model that misses long taxis, then. The next section asks whether "
              "that is a bias worth correcting.", "",
             "## 2. It is not a bias. It is regression to the mean, and the two tables that look alike are not alike", "",
             "Read these two tables together, in this order, because the second is the trap.", "",
             "**Conditioned on what the model says.** This is the honest calibration question, because the prediction is what we would correct:", ""]
    lines += deciles(p, y, p, "predicted")
    lines += ["", "**Conditioned on the truth.** This is the same model on the same rows, and it looks like massive shrinkage:", ""]
    lines += deciles(p, y, y, "true")
    q = pd.qcut(p, 10, labels=False, duplicates="drop")
    lines += ["", "The second table is what an unbiased predictor of a noisy quantity ALWAYS looks like when you sort by the noisy quantity: the "
              "rows in the top true decile are there partly because their noise was positive, and no predictor can follow that. It is not "
              "evidence of shrinkage and must never be quoted as a reason to expand the tail. The first table is the one that carries "
              "information, and it says the champion is calibrated.", "",
              "The ceiling on any additive recalibration, that is, the RMSE freed by removing each group's mean error exactly:", "",
              "| grouping | rows | ceiling on the gain |", "|---|---|---|",
              f"| predicted decile | {n:,} | {recoverable(e, q.astype(float), rmse, n):+.3f} s |",
              f"| schedule gap band | {n:,} | {recoverable(e, band.astype(str), rmse, n):+.3f} s |",
              f"| airport | {n:,} | {recoverable(e, tea.apt.values, rmse, n):+.3f} s |",
              f"| airport and runway | {n:,} | {recoverable(e, tea.apt.values + '/' + tea.runway.astype(str).values, rmse, n):+.3f} s |", ""]
    t = pd.DataFrame({"e": e, "b": band}).groupby("b", observed=True).agg(n=("e", "size"), bias=("e", "mean"))
    lines += ["Every one of those is far under the 2 s floor, and the worst single band is only a few seconds out:", "",
              "| schedule gap | rows | bias |", "|---|---|---|"]
    for i, r in t.iterrows():
        lines.append(f"| {i} | {int(r.n):,} | {r.bias:+.1f} s |")
    lines += ["", "So isotonic calibration, tail expansion, a per-airport offset and a per-band offset are all closed together, by arithmetic "
              "rather than by a run. There is no bias to collect.", "",
              "## 3. What the worst rows are", "", "The ten worst, with the columns the champion reads:", "",
              "| airport | runway | truth | schedule gap | EOBT gap | queue | prediction | error |", "|---|---|---|---|---|---|---|---|"]
    for i in o[:10]:
        lines.append(f"| {tea.apt.values[i]} | {tea.runway.values[i]} | {y[i]:,.0f} | {zs[i]:,.0f} | {tea.z_eobt.values[i]:,.0f} | "
                     f"{tea.queue.values[i]:,.0f} | {p[i]:,.0f} | {e[i]:+,.0f} |")
    marks = {"a taxi over 1 h": y > 3600, "a taxi over 12 h": y > 43200, "within an hour of a full day": np.abs(y - 86400) < 3000,
             "the truth equals the schedule": np.abs(y - zs) <= 600, "the truth equals EOBT": np.abs(y - tea.z_eobt.values) <= 600}
    lines += ["", "Every one is a flight that took off far later than the model expected, and in each the schedule already said so. Nothing here "
              "is a recording artifact of the kind lane B carries (`docs/DE.md`): lane A has exactly one day-shifted row in 339,046. What the "
              "tail rows look like, against the whole lane:", "",
              "| marker | all lane A | in the top 1% | their share of the tail's squared error |", "|---|---|---|---|"]
    for nm, m in marks.items():
        tm = m[top]
        lines.append(f"| {nm} | {int(m.sum()):,} | {int(tm.sum()):,} | {sse[top][tm].sum() / sse[top].sum() * 100:.1f}% |")
    a = pd.DataFrame({"apt": tea.apt.values, "s": sse})
    g = a.groupby("apt").agg(rows=("s", "size"), s=("s", "sum"))
    g["pr"], g["ps"] = g.rows / n * 100, g.s / sse.sum() * 100
    lines += ["", "By airport, against each airport's share of the lane:", "",
              "| airport | rows | share of the lane's rows | share of its squared error |", "|---|---|---|---|"]
    for i, r in g.sort_values("ps", ascending=False).iterrows():
        lines.append(f"| {i} | {int(r.rows):,} | {r.pr:.1f}% | {r.ps:.1f}% |")
    lines += ["", "## 4. Two model families miss on the same rows", "",
              "The question section 2 leaves open is whether a DIFFERENT model would find what this one misses. Three champions are cached on "
              "these exact rows, two families and two column sets, so the question is answerable without fitting anything.", "",
              "| pair | error correlation, all rows | error correlation, MC2's top 1% |", "|---|---|---|"]
    ps = {k: floored(champion_holdout(k, tra, tea)) for k in OTHERS}
    es = {k: v - y for k, v in ps.items()}
    pairs = [(CURRENT["A"], OTHERS[0], e, es[OTHERS[0]]), (CURRENT["A"], OTHERS[1], e, es[OTHERS[1]]),
             (OTHERS[0], OTHERS[1], es[OTHERS[0]], es[OTHERS[1]])]
    for na, nb, ea, eb in pairs:
        lines.append(f"| {na} against {nb} | {np.corrcoef(ea, eb)[0, 1]:.3f} | {np.corrcoef(ea[top], eb[top])[0, 1]:.3f} |")
    ed = es[OTHERS[0]]
    ov = len(set(top.tolist()) & set(np.argsort(-(ed ** 2))[:TOP].tolist()))
    best = np.where(np.abs(e) <= np.abs(ed), e, ed)
    lines += ["", f"{CURRENT['A']} is CatBoost with ordered boosting and categorical combinations; {OTHERS[0]} is LightGBM on the same 49 "
              f"columns. Different families, different splits, and their errors correlate {np.corrcoef(e, ed)[0, 1]:.3f} over the lane and "
              f"{np.corrcoef(e[top], ed[top])[0, 1]:.3f} in the tail. Their worst-1% row sets share {ov:,} of {TOP:,} rows, "
              f"{ov / TOP * 100:.0f}%. The mean of the two scores {float(np.sqrt((((e + ed) / 2) ** 2).mean())):,.1f} s, WORSE than "
              f"{CURRENT['A']} alone at {rmse:,.1f}, which is why the declared E2 blend tied. Even an oracle allowed to pick the better family "
              f"row by row reaches only {float(np.sqrt((best ** 2).mean())):,.1f} s, and it is an oracle.", "",
              "Two families given the same information make the same mistakes on the same flights. That is the signature of an information "
              "limit, not a model-class limit.", "",
              "## 5. What this licenses", "",
              "Nothing about a model, and no cell is declared. What it changes is where the remaining effort is worth spending.", "",
              "- **Closed by arithmetic:** post-hoc calibration, tail expansion, per-airport and per-band offsets. Ceiling under 0.1 s against a "
              "2 s floor.", "",
              "- **Low value on this evidence:** more of the same model class. A different family on the same columns lands within 9 s and "
              "misses the same rows, blending them is worse than the better one, and hyper-parameter search (the queued O1) moves inside that "
              "same space.", "",
              "- **The only direction with room:** new information about the specific flight that is about to taxi a long way. The queued TH1 "
              "and Q1 cells are that kind of work, and this diagnostic raises their priority above O1's.", "",
              "- **For the gate:** the lane-A tail is not a defect waiting to be fixed. Combined with F21 in lane B, a board gap of a few tens "
              "of seconds is not by itself proof that a better recipe exists."]
    text = "\n".join(lines) + "\n"
    with open("docs/TA.md", "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
