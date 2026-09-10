"""DE (diagnostic, never a verdict): the largest single block of error in the campaign is a block time recorded one day early.

F17 says nine holdout rows carry 34.4% of the composite squared error and LK1 says no ranking-visible column separates them. Neither
names the mechanism or prices a response to it. This does both: it identifies the artifact exactly, measures the ceiling of knowing
which rows carry it, and then prices every blind answer available to a model that cannot see them.
"""
from __future__ import annotations

import glob

import duckdb
import numpy as np
import pandas as pd

from taxi.cb1 import lane_b
from taxi.champ import CURRENT, lane_a, predict_b
from taxi.data import CV_FOLDS
from taxi.l2 import floored
from taxi.verdict import paired_delta, rmse

DAY = 86400.0  # one day: the recorded block time carries the right time of day and the wrong date
TOL = 3000.0  # a taxi time either side of the day; the lane's own p90 is under 1,500 s
BIG = 43200.0
HI = 40000.0  # a champion prediction this high only happens where the copy mixture has fired
RANK = "data/ranking.parquet"
ARR = """SELECT ADES_mvt AS apt, count(*) AS n, sum((TAXITIME_SEC_mvt > 3600)::INT) AS over_1h,
  sum((abs(TAXITIME_SEC_mvt - {day}) < {tol})::INT) AS day_early
  FROM read_parquet({files}) WHERE PHASE_mvt = 'ARR' AND TAXITIME_SEC_mvt IS NOT NULL GROUP BY 1"""


def families(d: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """big / copied-from-the-schedule (y = z_sched) / shifted by a day (y = 86400 + a taxi time)."""
    y, z = d.y.values, d.z_sched.values
    big = y > BIG
    return big, big & (np.abs(y - z) <= 3600), np.abs(y - DAY) < TOL


def _files(paths: list[str]) -> str:
    return "[" + ",".join(f"'{p.replace(chr(92), '/')}'" for p in paths) + "]"


def taxonomy(trb: pd.DataFrame, teb: pd.DataFrame, p: np.ndarray) -> list[str]:
    e = p - teb.y.values
    out = ["| set | rows | normal (under 1 h) | stale, 1 to 12 h | over 12 h | of those, day-shifted |", "|---|---|---|---|---|---|"]
    for tag, d in (("lane B train", trb), ("lane B holdout", teb)):
        y = d.y.values
        big, _, sh = families(d)
        out.append(f"| {tag} | {len(d):,} | {int((y < 3600).sum()):,} | {int(((y >= 3600) & ~big).sum())} | {int(big.sum())} | "
                   f"{int(sh.sum())} |")
    out += ["", "Where the champion's holdout squared error sits, by the same bands:", "",
            "| band | rows | share of lane-B squared error | mean prediction | mean truth |", "|---|---|---|---|---|"]
    y = teb.y.values
    for lo, hi, nm in ((0, 3600, "normal"), (3600, BIG, "stale, 1 to 12 h"), (BIG, 1e12, "over 12 h")):
        m = (y >= lo) & (y < hi)
        out.append(f"| {nm} | {int(m.sum()):,} | {(e[m] ** 2).sum() / (e ** 2).sum() * 100:.1f}% | {p[m].mean():,.0f} s | "
                   f"{y[m].mean():,.0f} s |")
    return out


PBANDS = ((0, 900), (900, 1800), (1800, 3600), (3600, 7200), (7200, BIG), (BIG, 1e12))


def reachable(teb: pd.DataFrame, p: np.ndarray, big: np.ndarray) -> list[str]:
    """The same squared error, banded on the model's own PREDICTION rather than on the truth.

    The table above bands on y, which no model can see at predict time. A band defined on the prediction IS reachable: if one
    of them held a systematic offset, a per-band constant would collect it with no new feature and no new fit. `bias share`
    is n x mean(e) squared over that band's SSE, the fraction a per-band constant could remove. The drop-k rows below then
    ask the other half of the question: whether what is left is a structure or one row.
    """
    e = p - teb.y.values
    sse = float((e ** 2).sum())
    out = ["| band, on the prediction | rows | share of lane-B squared error | mean signed error | bias share of the band |",
           "|---|---|---|---|---|"]
    for lo, hi in PBANDS:
        m = (p >= lo) & (p < hi)
        if not m.any():
            continue
        eb = e[m]
        b = float(len(eb) * eb.mean() ** 2 / (eb ** 2).sum())
        nm = f"{lo:,.0f} to {hi:,.0f} s" if hi < 1e11 else f"over {lo:,.0f} s"
        out.append(f"| {nm} | {int(m.sum()):,} | {(eb ** 2).sum() / sse * 100:.1f}% | {eb.mean():+,.0f} s | {b * 100:.1f}% |")
    q = p.copy()
    for lo, hi in PBANDS:
        m = (p >= lo) & (p < hi)
        if m.any():
            q[m] = p[m] - e[m].mean()
    v = paired_delta(e, floored(q) - teb.y.values, teb.day.values, lane="B")
    out += ["", f"Two bands are not flat. Both are tiny and neither is a lane, and the way to see that is to grant the whole "
            f"table its own constants: subtract each band's OWN holdout mean error, fitted on the rows it is scored on. That is "
            f"an oracle, not a model, so it can only be an upper bound. It takes lane B from {rmse(e):,.1f} s to "
            f"{rmse(floored(q) - teb.y.values):,.1f} s, {v.delta:+,.1f} [{v.lo:+,.1f}, {v.hi:+,.1f}] against a promotion floor "
            f"of {v.floor:.0f} s. A real per-band constant, fitted on the training months, would have to beat that ceiling from "
            "below.", ""]
    o = np.argsort(-(e ** 2))
    out += ["And the concentration, by dropping the worst rows one at a time. Nothing is predicted here; the rows are removed "
            "from the score to show what carries it.", "",
            "| lane-B holdout | rows | RMSE | share of the lane's squared error dropped | of the dropped rows, over 12 h |",
            "|---|---|---|---|---|", f"| all rows | {len(e):,} | {rmse(e):,.1f} | | |"]
    for k in (1, 5, 10, 50):
        keep = np.ones(len(e), bool)
        keep[o[:k]] = False
        out.append(f"| worst {k} dropped | {int(keep.sum()):,} | {rmse(e[keep]):,.1f} | "
                   f"{(1 - (e[keep] ** 2).sum() / sse) * 100:.1f}% | {int(big[o[:k]].sum())} of {k} |")
    i = o[0]
    out += ["", f"The lane is not a population, it is a handful of rows. One row of {len(e):,} carries "
            f"{(e[i] ** 2) / sse * 100:.1f}% of the squared error ({teb.apt.values[i]}, prediction {p[i]:,.0f} s against a "
            f"truth of {teb.y.values[i]:,.0f} s), the worst five carry "
            f"{(1 - (e[o[5:]] ** 2).sum() / sse) * 100:.1f}%, and "
            f"{int(big[o[:10]].sum())} of the worst ten are the over-12-hour rows section 2 tabulates one by one. So the "
            "concentration is not a new structure to search: it IS the artifact family, which section 4 prices and closes. "
            f"Below it, the remaining {int((~big).sum()):,} rows score {rmse(e[~big]):,.1f} s and the band table above says "
            "their error is spread, not offset.", ""]
    return out


def cv_mixture(trb: pd.DataFrame) -> list[str]:
    """The only blind answer with any structure: split the champion's own high predictions between the two artifact values."""
    out = ["| fold | eval rows | predictions over 40,000 s | fit copies | fit day-shifts | q | champion | mixed | gain |",
           "|---|---|---|---|---|---|---|---|---|"]
    keep = []
    for fold in CV_FOLDS:
        m = trb.month.isin(fold).values
        fit, ev = trb[~m].reset_index(drop=True), trb[m].reset_index(drop=True)
        p = floored(predict_b(CURRENT["B"], fit, ev))
        _, cp, sh = families(fit)
        ncp, nsh = int(cp.sum()), int(sh.sum())
        q = ncp / max(ncp + nsh, 1)
        hi = p > HI
        p2 = p.copy()
        p2[hi] = q * p[hi] + (1 - q) * (DAY + float(np.median(fit.y.values[fit.y.values < 3600])))
        r1, r2 = rmse(p - ev.y.values), rmse(p2 - ev.y.values)
        keep.append((p - ev.y.values, p2 - ev.y.values))
        out.append(f"| {fold} | {len(ev):,} | {int(hi.sum())} | {ncp} | {nsh} | {q:.3f} | {r1:.1f} | {r2:.1f} | {r1 - r2:+.1f} |")
    a, b = (np.concatenate([k[i] for k in keep]) for i in (0, 1))
    out.append(f"| pooled | {len(a):,} | | | | | {rmse(a):.1f} | {rmse(b):.1f} | {rmse(a) - rmse(b):+.1f} |")
    return out


def main() -> None:
    trb, teb = lane_b()
    _, tea = lane_a()
    p, y = floored(predict_b(CURRENT["B"], trb, teb)), teb.y.values
    big, cp, sh = families(teb)
    _, _, sh_tr = families(trb)
    normal = float(np.median(trb.y.values[trb.y.values < 3600]))
    rate = float(sh_tr.mean())

    lines = ["# DE: the block time recorded one day early (diagnostic, never a verdict)", "",
             f"Regenerate with `python -m taxi.de`. Lane-B champion {CURRENT['B']}, floored, holdout Jan + Jul 2025. Nothing here decides a "
             "verdict and no cell is declared from it: every answer it prices either loses or lands under the 2 s floor.", "",
             "## 1. The artifact, exactly", "",
             "A lane-B row whose taxi time sits within an hour of a full day is not a long taxi. Its recorded block time carries the right time "
             f"of day and the wrong date. Subtract the day and what is left is an ordinary taxi time (the lane's own median is {normal:,.0f} s):",
             "", "| set | y minus 86,400, one row each |", "|---|---|"]
    for tag, d in (("lane B train", trb), ("lane B holdout", teb)):
        _, _, s = families(d)
        lines.append(f"| {tag} | {', '.join(f'{v:,.0f}' for v in np.sort(d.y.values[s] - DAY))} |")
    lines += ["", "That is the whole mechanism. It is distinct from the schedule-copy artifact the L4 to L8 chain models, where the block time "
              "field was filled with the scheduled time and y equals z_sched. Both produce a taxi time of many hours; only the copy family is "
              "reachable from the features, because a copied row's z_sched IS its answer.", "",
              "## 2. How much error it carries", ""]
    lines += taxonomy(trb, teb, p)
    lines += ["", "### The same error, banded on what the model can actually see", "",
              "Bands on y are a post-mortem. A band on the model's own prediction is a lane: a per-band constant needs no new "
              "feature and no new fit, so if one of these bands held a systematic offset it would be free to collect.", ""]
    lines += reachable(teb, p, big)
    lines += ["", f"Of {int(big.sum())} holdout rows over 12 h, {int((big & cp).sum())} are schedule copies and {int((big & sh).sum())} are day-shifted. "
              "Every one of them, with the champion on it:", "",
              "| airport | day | truth | z_sched | prediction | family |", "|---|---|---|---|---|---|"]
    for i in np.argsort(-teb.y.values * big):
        if not big[i]:
            break
        fam = "day-shifted" if sh[i] else ("schedule copy" if cp[i] else "neither")
        lines.append(f"| {teb.apt.values[i]} | {teb.day.values[i]} | {teb.y.values[i]:,.0f} | {teb.z_sched.values[i]:,.0f} | "
                     f"{p[i]:,.0f} | {fam} |")
    lines += ["", "The two LFPG rows F17 names as holding 28.4% of the whole composite's squared error are in that table, and they are not the "
              "same thing as each other: one is day-shifted and one belongs to neither family.", "",
              "## 3. Where it lives, and why the 2026 arrivals cannot help", ""]
    con = duckdb.connect()
    tr_files = sorted(glob.glob("data/training_2025-*.parquet"))
    a25 = con.sql(ARR.format(day=DAY, tol=TOL, files=_files(tr_files))).df()
    a26 = con.sql(ARR.format(day=DAY, tol=TOL, files=_files([RANK]))).df()
    lines += ["The ranking file blanks the departure taxi time and keeps the arrival one, so an arrival is the only place a 2026 artifact rate "
              "could be read. It is not there.", "",
              "| set | rows | over an hour | day-shifted |", "|---|---|---|---|",
              f"| 2025 arrivals, all 12 months | {int(a25.n.sum()):,} | {int(a25.over_1h.sum())} | {int(a25.day_early.sum())} |",
              f"| 2026 arrivals, Jan + Jul (ranking) | {int(a26.n.sum()):,} | {int(a26.over_1h.sum())} | {int(a26.day_early.sum())} |",
              f"| 2025 lane-B departures, all 12 months | {len(trb) + len(teb):,} | | {int(sh_tr.sum()) + int(sh.sum())} |", "",
              "Two day-shifted arrivals in 2.08 M rows of 2025 and none at all in 2026, against fourteen in 22,470 lane-B departures. The "
              "artifact belongs to off-block recording on movements with no flight plan, so the arrivals carry no estimate of its 2026 rate.", "",
              "## 4. The ceiling", ""]
    o = p.copy()
    o[sh] = DAY + normal
    v = paired_delta(p - y, o - y, teb.day.values, lane="B")
    lines += [f"Predicting {DAY + normal:,.0f} s on exactly the {int(sh.sum())} day-shifted holdout rows and changing nothing else takes lane B "
              f"from {rmse(p - y):,.1f} s to {rmse(o - y):,.1f} s, **{v.delta:+,.1f} s** [{v.lo:+,.1f}, {v.hi:+,.1f}]. That is about "
              f"{v.delta * rmse(p - y) * len(teb) / (322.5796 * (len(teb) + len(tea))):.0f} s of composite, larger than the whole distance from "
              "our S1 board score to the leader. Six rows of 344,419. It is a ceiling and not a lane: the label is not visible in 2026.", "",
              "The per-row table says the ceiling is narrower still, and says why. On five of the six the SCHEDULE was shifted with the block "
              "time, so z_sched already carries the day and the champion lands within a few thousand seconds unaided. On one the schedule stayed "
              "normal and only the block time moved. Splitting the oracle along that line:", "",
              "| oracle applied to | rows | lane B RMSE | paired against the champion [95%] |", "|---|---|---|---|"]
    for nm, sub in (("the schedule moved too (z_sched over 40,000 s)", sh & (teb.z_sched.values > 40000)),
                    ("only the block time moved", sh & (teb.z_sched.values <= 40000))):
        z = p.copy()
        z[sub] = DAY + normal
        u = paired_delta(p - y, z - y, teb.day.values, lane="B")
        lines.append(f"| {nm} | {int(sub.sum())} | {rmse(z - y):,.1f} | {u.delta:+.1f} [{u.lo:+.1f}, {u.hi:+.1f}] |")
    lines += ["", "Most of the ceiling is one row, but not all of it, and the remainder is the part that matters. The single LFPG row is "
              "unreachable: its schedule is an ordinary 1,740 s and nothing marks it. The five LIRF rows are worth +85.5 s between them and they "
              "are NOT unreachable, because the thing that identifies them, a schedule already past 40,000 s, is a ranking-visible column. The "
              "champion under-predicts three of them badly (75,763 against 87,361, 60,069 against 87,186, 61,729 against 87,177) because it "
              "follows z_sched instead of the day. That looked like a lane. It is not, and the population says why:", ""]
    lines += ["| lane-B rows with z_sched over 40,000 s | rows | schedule copies | day-shifted | ordinary flights |", "|---|---|---|---|---|"]
    for tag, d in (("train", trb), ("holdout", teb)):
        m = d.z_sched.values > 40000
        yy, zz = d.y.values[m], d.z_sched.values[m]
        c = np.abs(yy - zz) <= 3600
        s = np.abs(yy - DAY) < TOL
        lines.append(f"| {tag} | {int(m.sum())} | {int(c.sum())} | {int(s.sum())} | {int((~c & ~s).sum())} |")
    m = teb.z_sched.values > 40000
    yy, zz = teb.y.values[m], teb.z_sched.values[m]
    ord_y = np.sort(yy[(np.abs(yy - zz) > 3600) & (np.abs(yy - DAY) >= TOL)])
    lines += ["", "A schedule already many hours out is the only thing these rows share, and it is mostly a stale schedule on an ordinary "
              f"flight. Of the {int(m.sum())} such holdout rows, {len(ord_y)} have perfectly normal taxi times "
              f"({', '.join(f'{v:,.0f}' for v in ord_y[:6])} s and so on). Any rule keyed on that signature pays about 86,000 s of error on each "
              "of those to win about 25,000 s on each of five. The champion already gets them right, which is what its classifier is for. No "
              "cell is declared: the lane was killed by its own population before it cost a verdict.", "",
              "## 5. Every blind answer, priced", "",
              "### 5a. A uniform shift at the training rate", "",
              f"The RMSE-optimal response to an artifact of magnitude A on a row with probability p is to add p x A. The training rate is "
              f"{int(sh_tr.sum())} of {len(trb):,} rows, {rate * 100:.4f}%, so the shift is {rate * DAY:,.1f} s on every lane-B prediction.", "",
              "| shift | lane B RMSE | paired against the champion [95%] |", "|---|---|---|"]
    for s in (rate * DAY, 2 * rate * DAY, 100.0):
        q = floored(p + s)
        w = paired_delta(p - y, q - y, teb.day.values, lane="B")
        lines.append(f"| {s:,.1f} s | {rmse(q - y):,.1f} | {w.delta:+.1f} [{w.lo:+.1f}, {w.hi:+.1f}] |")
    lines += ["", "The best of these lands under the 2 s floor, and the arithmetic says why it cannot do better. The gain from an optimal shift "
              "is n p squared A squared, so at a rate near one in two thousand the whole lane is worth a couple of seconds, and every row that "
              "is not an artifact pays the shift squared.", "",
              "### 5b. Splitting the champion's own high predictions between the two artifact values", "",
              "The one structured answer available. Where the champion already predicts over 40,000 s the copy mixture has fired, so the row is "
              "one artifact or the other; predict q x z_sched + (1 - q) x (86,400 + a normal taxi), with q the copy share of the fit months' "
              "big rows. Five month-pair folds:", ""]
    lines += cv_mixture(trb)
    hi = p > HI
    q = float((families(trb)[1]).sum()) / max(float((families(trb)[1]).sum() + sh_tr.sum()), 1.0)
    m2 = p.copy()
    m2[hi] = q * p[hi] + (1 - q) * (DAY + normal)
    w = paired_delta(p - y, m2 - y, teb.day.values, lane="B")
    lines += ["", f"It loses on CV and it loses on the holdout: at q = {q:.2f} the same recipe scores {rmse(m2 - y):,.1f} s against "
              f"{rmse(p - y):,.1f}, {w.delta:+.1f} [{w.lo:+.1f}, {w.hi:+.1f}]. The reason is in section 1: a day-shifted row need not have a "
              f"strange schedule at all. Of the {int(hi.sum())} holdout rows the champion pushes over 40,000 s, {int((hi & sh).sum())} are "
              f"day-shifted, so the split pays out on the wrong rows: {int((hi & ~sh).sum())} of them are not day-shifted at all and take "
              f"the pull toward 86,400 for nothing, while the {int((sh & ~hi).sum())} day-shifted row the champion leaves low is never "
              f"reached, its z_sched being an ordinary "
              f"{', '.join(f'{v:,.0f}' for v in np.sort(teb.z_sched.values[sh & ~hi]))} s.", "",
              "## 6. What this closes", "",
              "The largest single gain in the campaign is real, measured, and out of reach by every route tested. Knowing which six rows carry "
              "it is worth more than the distance from our board score to the leader. Nothing reaches them: the one row that holds most of the "
              "gain has an ordinary schedule and no mark on it; the five that are marked share their mark with twice as many ordinary flights "
              "and with the schedule copies, so a rule keyed on it loses more than it wins; the 2026 arrivals carry no day-shifted rows at all "
              "and cannot price the rate; a uniform shift at the measured rate lands under the 2 s floor by the n p squared A squared "
              "arithmetic; and the one structured split loses three of five folds, pools negative and loses on the holdout. Section 2 "
              "closes the last blind route with it: banded on the model's OWN prediction, which is the only banding a model can "
              "act on, the lane holds no offset worth collecting, and an oracle that fits a constant per band on the very rows "
              "it is scored on lands under the lane-B promotion floor with an interval through zero. Flags F21 and F43 carry "
              "this so no later report re-opens it as an opportunity, and so that a board gap of a few tens of seconds is not read as proof "
              "that a better recipe exists."]
    text = "\n".join(lines) + "\n"
    with open("docs/DE.md", "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
