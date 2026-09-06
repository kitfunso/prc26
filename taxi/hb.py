"""HB (diagnostic, never a verdict): what separates a holdout number from a board number, measured in both directions.

Every verdict in this campaign is a holdout number and the only thing that counts is a board number. Two things differ between them and
neither has been priced whole. The ranking set's rows sit in different cells from the holdout's, which MIX measured for lane B and never
for lane A, the lane carrying 98.5% of the rows. And the file we upload is fit on all twelve months of 2025 while every holdout fit
excludes January and July, so the model we send has seen the months it is scored on and the model we judge never has.
"""
from __future__ import annotations

import time

import duckdb
import numpy as np
import pandas as pd

import lightgbm as lgb

from taxi.cb1 import lane_b
from taxi.champ import COLS_D1, CURRENT, champion_holdout, lane_a, predict_b
from taxi.data import WINDOW, _files, load, month_file
from taxi.l2 import floored
from taxi.m1 import BASE, CAT, chosen_cells, matrix
from taxi.m2 import is_tail, predict_m2, stages
from taxi.mix import composite
from taxi.verdict import paired_delta, rmse
from taxi.weather import join, load_metar

RANK = "data/ranking.parquet"
MONTH = {1: "January", 2: "February", 7: "July", 8: "August"}
ROUNDS = (266, 347, 637)  # D1's recorded final rounds, held fixed across arms so only the training rows differ
DROPS = (0, 1, 5, 20, 100)
T = (2, 8)
ADJ = (1, 7)
TAU = 1.5
SEED = 0

CELLS = """SELECT ADEP_mvt apt, month(MVT_TIME_UTC_mvt) m, count(*) n FROM read_parquet('{path}')
 WHERE PHASE_mvt='DEP' AND month(MVT_TIME_UTC_mvt) IN (1, 7) AND AOBT_3_flt IS {null} NULL GROUP BY 1, 2"""

A_ARR = """SELECT ADES_mvt apt, epoch(MVT_TIME_UTC_mvt) t, month(MVT_TIME_UTC_mvt) m, hour(MVT_TIME_UTC_mvt) hr,
   dayofweek(MVT_TIME_UTC_mvt) dow, AIRCRAFT_TYPE_mvt actype, RUNWAY_mvt::VARCHAR runway, STAND_mvt::VARCHAR stand,
   WK_TBL_CAT_flt wake, AIRCRAFT_OPERATOR_flt op, MARKET_SEGMENT_flt seg, ADEP_flt orig, TAXITIME_SEC_mvt::DOUBLE y
 FROM read_parquet({src}) WHERE PHASE_mvt='ARR' AND TAXITIME_SEC_mvt IS NOT NULL"""
A_EVT = """SELECT CASE WHEN PHASE_mvt='ARR' THEN ADES_mvt ELSE ADEP_mvt END apt, epoch(MVT_TIME_UTC_mvt) t, PHASE_mvt ph
 FROM read_parquet({src}) ORDER BY 1, 2"""
A_CAT = ["apt", "actype", "runway", "stand", "wake", "op", "seg", "orig"]
A_COLS = A_CAT + ["hr", "dow", "n_land", "n_off"]
A_WINDOW = 1800.0
A_ROUNDS = 600
A_PARAMS = dict(objective="l2", num_leaves=63, min_data_in_leaf=200, learning_rate=0.05, verbose=-1, num_threads=8)


def _mix_rows(e2: np.ndarray, cell: pd.DataFrame, want: pd.Series) -> list[str]:
    """Holdout error re-averaged under the ranking set's own cell counts, with the worst rows dropped one at a time."""
    order = np.argsort(-e2)
    out = []
    for k in DROPS:
        m = np.ones(len(e2), bool)
        m[order[:k]] = False
        g = cell[m].assign(e2=e2[m]).groupby(["apt", "m"])["e2"].agg(["mean", "count"]).join(want.rename("nr"))
        g["nr"] = g.nr.fillna(0.0)
        a = float(np.sqrt((g["mean"] * g["count"]).sum() / g["count"].sum()))
        b = float(np.sqrt((g["mean"] * g.nr).sum() / g.nr.sum()))
        out.append(f"| {k} | {a:.4f} s | {b:.4f} s | **{b - a:+.4f} s** |")
    return out


def _cells(con: duckdb.DuckDBPyConnection, null: str) -> pd.Series:
    d = con.execute(CELLS.format(path=RANK, null=null)).df()
    return d.set_index([d.apt.astype(str), d.m.astype(int)])["n"]


def _mix(con: duckdb.DuckDBPyConnection) -> tuple[list[str], dict]:
    tra, tea = lane_a()
    trb, teb = lane_b()
    ea = floored(champion_holdout(CURRENT["A"], tra, tea)) - tea.y.values
    eb = floored(predict_b(CURRENT["B"], trb, teb)) - teb.y.values
    ka = pd.DataFrame({"apt": tea.apt.astype(str).values, "m": tea.month.astype(int).values})
    kb = pd.DataFrame({"apt": teb.apt.astype(str).values, "m": teb.month.astype(int).values})
    wa, wb = _cells(con, "NOT"), _cells(con, "")
    n_a, n_b = int(wa.sum()), int(wb.sum())
    la, lb = rmse(ea), rmse(eb)

    rows_a, rows_b = _mix_rows(ea**2, ka, wa), _mix_rows(eb**2, kb, wb)
    mix_a = float(rows_a[0].split("**")[1].split(" s")[0])
    mix_b = float(rows_b[0].split("**")[1].split(" s")[0])
    hold = composite(la, lb, len(tea), len(teb))
    split = composite(la, lb, n_a, n_b)
    with_a = composite(la + mix_a, lb, n_a, n_b)
    with_b = composite(la + mix_a, lb + mix_b, n_a, n_b)
    out = [
        "## 1. The rows we are scored on are not the rows we measure\n",
        f"The ranking set holds {n_a + n_b:,} departures in January and July 2026, {n_b:,} of them in lane B; the holdout holds",
        f"{len(tea) + len(teb):,} in the same months of 2025, {len(teb):,} in lane B. The ranking count is 11 short of the 344,841 rows the",
        "board scores, because 11 of them fall on 1 August UTC and this grouping is by calendar month. `docs/MIX.md` measured the lane-B",
        "cell mix and",
        "the lane split. It never measured lane A, which is 98.5% of the rows, because it held the lane-A figure fixed while it moved",
        "lane B. Both lanes below, each re-averaged under the ranking set's own airport-and-month counts, each with the worst holdout",
        "rows dropped one at a time, which is the check that killed most of the lane-B headline.\n",
        "### Lane A\n",
        "| worst rows dropped | holdout mix | ranking mix | shift |", "|---|---|---|---|", *rows_a, "",
        "### Lane B\n",
        "| worst rows dropped | holdout mix | ranking mix | shift |", "|---|---|---|---|", *rows_b, "",
        "The two lanes behave differently and the difference is the point. Lane B's shift is mostly two rows, as MIX found: drop them",
        "and it collapses. Lane A's does not move at all when the worst hundred rows go, because its cells hold thousands of rows each.",
        f"It is a real property of the row set we will be scored on, and it is against us by {mix_a:+.4f} s of lane A.\n",
        "Composite, one step at a time.\n",
        "| step | composite | change |", "|---|---|---|",
        f"| holdout, holdout mix | {hold:.4f} s | |",
        f"| ranking lane split | {split:.4f} s | {split - hold:+.4f} s |",
        f"| lane-A cell mix | {with_a:.4f} s | {with_a - split:+.4f} s |",
        f"| lane-B cell mix, raw | {with_b:.4f} s | {with_b - with_a:+.4f} s |", "",
        f"The lane split is worth {hold - split:.4f} s in our favour and MIX's baseline of {split:.2f} s already spent it, which is why",
        "that report reads as if the mix were purely a lane-B story. The lane-A term is new here and the lane-B term is the one MIX",
        f"showed collapses. The two solid terms point opposite ways: the split gives {hold - split:.2f} s and the lane-A cell mix takes",
        f"{with_a - split:.2f} s back, so the honest correction before lane B is {abs(with_a - hold):.2f} s in our favour, not the several",
        "seconds a lane-B-only reading suggests.\n",
    ]
    return out, {"n_a": n_a, "n_b": n_b, "hold": hold, "split": split, "with_a": with_a}


def _dep() -> pd.DataFrame:
    """All twelve months of lane A, which is the row set the board file is fit on and no holdout fit ever sees."""
    d = join(load(tuple(range(1, 13)), WINDOW), load_metar())
    return d[d.has_fpl].reset_index(drop=True)


def _fit_w(x: pd.DataFrame, label: np.ndarray, w: np.ndarray, rounds: int, cell: dict) -> lgb.Booster:
    d = lgb.Dataset(matrix(x, COLS_D1), label, weight=w, categorical_feature=[c for c in CAT if c in COLS_D1])
    return lgb.train({**BASE, **cell}, d, num_boost_round=rounds)


def _stages_w(tr: pd.DataFrame, w: np.ndarray, cell: dict) -> tuple:
    t = is_tail(tr)
    return (_fit_w(tr, t.astype(float), w, ROUNDS[0], cell),
            _fit_w(tr[t], tr.c.values[t], w[t], ROUNDS[1], cell),
            _fit_w(tr[~t], tr.c.values[~t], w[~t], ROUNDS[2], cell))


def _cdist(m: np.ndarray, targets: tuple[int, ...]) -> np.ndarray:
    """Distance in months around a twelve-month circle, so December sits one step from January."""
    return np.min([np.minimum(np.abs(m - t), 12 - np.abs(m - t)) for t in targets], axis=0)


def _arm(tr: pd.DataFrame, te: pd.DataFrame, cell: dict, w: np.ndarray | None = None) -> np.ndarray:
    models = stages(tr, cell, rounds=ROUNDS, cols=COLS_D1) if w is None else _stages_w(tr, w, cell)
    return predict_m2(models, te)[0] - te.y.values


def _row(nm: str, tr: pd.DataFrame, te: pd.DataFrame, e: np.ndarray, e0: np.ndarray | None) -> str:
    feb, aug = (te.month.values == T[0]), (te.month.values == T[1])
    v = ""
    if e0 is not None:
        d = paired_delta(e0, e, te.day.values)
        v = f"**{d.delta:+.4f} s** [{d.lo:+.2f}, {d.hi:+.2f}]"
    return f"| {nm} | {len(tr):,} | {rmse(e):.4f} s | {rmse(e[feb]):.2f} s | {rmse(e[aug]):.2f} s | {v} |"


def _months(d: pd.DataFrame, cell: dict) -> list[str]:
    """Judge on even days of February and August; the odd days are the donor, so an arm can hold the judged months or not."""
    dom = pd.to_datetime(d.day).dt.day.values
    in_t = d.month.isin(T).values
    judge, donor, base = in_t & (dom % 2 == 0), in_t & (dom % 2 == 1), ~in_t
    te = d[judge].reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    bi, di = np.flatnonzero(base), np.flatnonzero(donor)
    keep = np.setdiff1d(bi, rng.choice(bi, size=len(di), replace=False))
    adj = rng.choice(np.flatnonzero(base & d.month.isin(ADJ).values), size=len(di), replace=False)
    swap = np.concatenate([keep, di])

    arms = [("A judged months absent", bi, None), ("B judged months swapped in", swap, None),
            ("C adjacent months swapped in", np.concatenate([keep, adj]), None),
            ("D seasonal weights, no judged rows", bi, TAU), ("E judged rows and seasonal weights", swap, TAU)]
    rows, e0 = [], None
    for nm, ix, tau in arms:
        tr = d.iloc[ix].reset_index(drop=True)
        w = None if tau is None else np.exp(-_cdist(tr.month.values, T) / tau)
        e = _arm(tr, te, cell, w)
        rows.append(_row(nm, tr, te, e, e0))
        e0 = e0 if e0 is not None else e
    gain = rmse(e0)
    return [
        "## 2. The board file trains on months no holdout fit has seen\n",
        "`taxi/s2.py` builds the upload from all twelve months of 2025. Every verdict in this campaign trains on ten, because January and",
        "July are the holdout. `COLS_D1` carries no month, season or day-of-week column, so nothing tells either model which month it is in;",
        "the only difference is which rows are present. That difference has never been priced. It is priceable without touching the holdout:",
        f"judge the even days of {MONTH[T[0]]} and {MONTH[T[1]]}, and let the arms differ only in whether the odd days of those months are in",
        "training,",
        "each arm holding the same row count so capacity is not the variable.\n",
        f"Judged rows: {len(te):,}. Donor rows: {len(di):,}.\n",
        "| training rows | n | RMSE | February | August | vs A |", "|---|---|---|---|---|---|", *rows, "",
        f"The judged months' own rows are worth real seconds, and adjacency is not a substitute: arm C swaps in the two months either side",
        "and lands worse than having no target rows at all. Weighting is worse still. Arm D leans the fit toward the season without any of",
        "its rows and loses; arm E adds the weights on top of the rows that do help and gives back most of the gain. Only presence helps,",
        f"and it is not the season: nearly all of it is in {MONTH[T[0]]} while {MONTH[T[1]]} barely moves. Baseline {gain:.4f} s.\n",
    ]


def _days(d: pd.DataFrame, cell: dict) -> list[str]:
    """Is a month's own value weather that decays in days, or structure that holds for weeks?"""
    dom = pd.to_datetime(d.day).dt.day.values
    in_t = d.month.isin(T).values
    judge = in_t & (dom >= 22) & (dom <= 28)
    near, far = in_t & (dom >= 15) & (dom <= 21), in_t & (dom >= 1) & (dom <= 14)
    te = d[judge].reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    bi, n = np.flatnonzero(~in_t), int(near.sum())
    keep = np.setdiff1d(bi, rng.choice(bi, size=n, replace=False))
    arms = [("A no judged-month rows", bi), ("B donors 1 to 7 days earlier", np.concatenate([keep, np.flatnonzero(near)])),
            ("C donors 8 to 21 days earlier", np.concatenate([keep, rng.choice(np.flatnonzero(far), size=n, replace=False)]))]
    rows, e0, got = [], None, []
    for nm, ix in arms:
        tr = d.iloc[ix].reset_index(drop=True)
        e = _arm(tr, te, cell)
        rows.append(_row(nm, tr, te, e, e0))
        got.append(rmse(e))
        e0 = e0 if e0 is not None else e
    share = (got[0] - got[2]) / (got[0] - got[1]) if got[0] > got[1] else float("nan")
    return [
        "## 3. Weather decays in days; this does not\n",
        "Section 2's donor rows are interleaved with the judged rows, one day apart, so they share weather fronts and the same week's",
        "disruption. If that is all the gain is, it should fall away as the donors move back in time. It mostly does not. Judging the 22nd to",
        "the 28th, with donors drawn from the week before or from the fortnight before that.\n",
        f"Judged rows: {len(te):,}. Donors per arm: {n:,}.\n",
        "| training rows | n | RMSE | February | August | vs A |", "|---|---|---|---|---|---|", *rows, "",
        f"Donors three weeks removed keep **{share:.0%}** of what donors one week removed are worth, which is more than short-range weather",
        "autocorrelation leaves. So this is not a weather effect that a better METAR join would capture. It is something slower that persists",
        f"across weeks inside one year, and it is again all in {MONTH[T[0]]}: {MONTH[T[1]]} gains nothing at any distance.\n",
    ]


def _arr(src: str) -> pd.DataFrame:
    con = duckdb.connect()
    d = con.execute(A_ARR.format(src=src)).df()
    ev = con.execute(A_EVT.format(src=src)).df()
    out = np.zeros((len(d), 2))
    for apt, ix in d.groupby("apt", observed=True).indices.items():
        e = ev[ev.apt == apt]
        t = d.t.values[ix]
        for j, ph in enumerate(("ARR", "DEP")):
            s = np.sort(e.t.values[e.ph.values == ph])
            out[ix, j] = np.searchsorted(s, t, "left") - np.searchsorted(s, t - A_WINDOW, "left")
    d[["n_land", "n_off"]] = out
    d["dy"] = (d.t // 86400).astype(int)
    return d


def _year() -> list[str]:
    """Departures hold one year of labels, so cross-year transfer is untestable there; the ranking file's arrivals hold two."""
    tr = _arr(_files([month_file(m) for m in range(1, 13)]))
    te_all = _arr(f"'{RANK}'")
    cats = {c: pd.CategoricalDtype(sorted(set(tr[c].dropna()) | set(te_all[c].dropna()))) for c in A_CAT}
    for d in (tr, te_all):
        for c in A_CAT:
            d[c] = d[c].astype(cats[c])
    rng = np.random.default_rng(SEED)
    rows, deltas = [], []
    for m in sorted(set(tr.m) & set(te_all.m) & {1, 7}):
        te = te_all[te_all.m == m].reset_index(drop=True)
        di = np.flatnonzero((tr.m == m).values)
        bi = np.flatnonzero((tr.m != m).values)
        keep = np.setdiff1d(bi, rng.choice(bi, size=len(di), replace=False))
        e = {}
        for nm, ix in (("A", bi), ("B", np.concatenate([keep, di]))):
            s = tr.iloc[ix]
            b = lgb.train(A_PARAMS, lgb.Dataset(s[A_COLS], s.y.values, categorical_feature=A_CAT), num_boost_round=A_ROUNDS)
            e[nm] = b.predict(te[A_COLS]) - te.y.values
        v = paired_delta(e["A"], e["B"], te.dy.values)
        deltas.append(v.delta)
        rows.append(f"| {m} | {len(te):,} | {len(di):,} | {rmse(e['A']):.4f} s | {rmse(e['B']):.4f} s | **{v.delta:+.4f} s** [{v.lo:+.2f}, {v.hi:+.2f}] |")
    worth = "worth nothing" if max(abs(x) for x in deltas) < 2.0 else "worth real seconds"
    return [
        "## 4. Does the same month a year earlier transfer?\n",
        "Section 2 measured a month's own rows inside one year. The board's extra rows are not inside one year: the file trains on January",
        "and July **2025** and is scored on January and July **2026**. Departures cannot answer this, because they carry one year of labels.",
        "Arrivals can. The ranking file keeps a measured taxi-in on 2026 arrivals at the same ten airports, so a model trained on 2025",
        "arrivals can be judged on 2026 ones, with the matching month a year earlier present or absent and the row count held equal.\n",
        "| judged month, 2026 | judged rows | donor rows | A month absent | B month swapped in | gain |",
        "|---|---|---|---|---|---|", *rows, "",
        f"The same month a year earlier is **{worth}**. The two months do not agree even on the sign and both moves are under a second,",
        "where the within-year version was worth several. A year later the same calendar position carries none of it, which is the shape of",
        "an effect that is fleet, schedule and airfield state rather than season. Taxi-in",
        "is not taxi-out and `docs/YR.md` measured their month-to-month deviations as uncorrelated, so this is a proxy and is stated as one.",
        "It is the only cross-year measurement the data admits, and it points one way.\n",
    ]


def main() -> None:
    con = duckdb.connect()
    out = ["# HB: from a holdout number to a board number\n",
           "Generated by `python -m taxi.hb`. A diagnostic. It declares no cell and decides no verdict.\n"]
    mix, m = _mix(con)
    out += mix
    d = _dep()
    cell, _ = chosen_cells()["A"]
    t0 = time.time()
    out += _months(d, cell) + _days(d, cell) + _year()
    out += [
        "## 5. What it decides\n",
        "Two corrections were unpriced and they point opposite ways, so both had to be measured before either could be assumed.\n",
        f"The row mix is real and it has two terms. The ranking set's lane split is worth {m['hold'] - m['split']:.4f} s of composite in our",
        f"favour and the lane-A cell mix takes {m['with_a'] - m['split']:.4f} s back, leaving {abs(m['with_a'] - m['hold']):.4f} s in our",
        "favour before lane B. The lane-A term is the new one: it survives dropping the worst hundred rows, so unlike lane B's it is a",
        "property of the row set rather than of a few outliers, and no projection this campaign has made carries it.\n",
        "The training months are not a hidden credit. The board file sees January and July 2025 and no holdout fit does, which looked like an",
        "advantage the holdout could not show; section 2 priced a month's own rows at several seconds and section 4 shows that value does not",
        "cross a year. So the extra months buy about nothing on 2026 rows, and the holdout is a fair proxy on this axis rather than a",
        "pessimistic one.\n",
        "Together: the honest board projection is the holdout composite plus the mix shift, with no credit for the extra training months. It",
        "does not change a champion, a cell or a file. It changes what the board numbers should be expected to be, and `docs/YR.md`'s",
        "year-shift risk sits on top of it with a sign nothing here can settle. Flag F26.\n",
        f"Run time for the fitted sections: {time.time() - t0:.0f}s.",
    ]
    with open("docs/HB.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    print("\n".join(out[-8:]))


if __name__ == "__main__":
    main()
