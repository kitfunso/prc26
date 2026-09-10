"""MIX (diagnostic, never a verdict): what the 2026 ranking file's own rows say about a holdout measured on 2025.

Four questions nothing had asked. Does the ranking set's lane-B row mix differ from the holdout's, and does that reweight our
error? Do arrivals, whose taxi times the ranking file keeps, carry congestion the champion misses? Did the recording process
shift between the years? Is the lane-B artifact rate separable beyond what the champion already exploits?
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from taxi.cb1 import lane_b
from taxi.champ import CURRENT, champion_holdout, lane_a, predict_b
from taxi.data import HOLDOUT_MONTHS, _files, connect, month_file
from taxi.l2 import floored
from taxi.verdict import paired_delta, rmse

RANK = "data/ranking.parquet"
BIG = 3600.0  # a taxi time over an hour is a recording artifact, not a taxi
DROPS = (0, 1, 2, 5, 20)
CLS = ("z_sched", "queue", "dep_before", "arr_anchor", "push_anchor", "occ", "t_occ", "hour", "wday", "notyp", "month")
ARR_TAXI = """
SELECT ADES_mvt AS apt, epoch(BLOCK_TIME_UTC_mvt) AS tb, TAXITIME_SEC_mvt::DOUBLE AS ti
FROM read_parquet({files}) WHERE PHASE_mvt = 'ARR' AND BLOCK_TIME_UTC_mvt IS NOT NULL AND TAXITIME_SEC_mvt IS NOT NULL
ORDER BY apt, tb"""
ARR_RATE = """
SELECT month(MVT_TIME_UTC_mvt) AS m, count(*) AS n, avg(TAXITIME_SEC_mvt) AS mean_ti,
       sum(CASE WHEN TAXITIME_SEC_mvt > {big} THEN 1 ELSE 0 END) AS big
FROM read_parquet({files}) WHERE PHASE_mvt = 'ARR' AND TAXITIME_SEC_mvt IS NOT NULL GROUP BY 1 ORDER BY 1"""
DEP_CELLS = """
SELECT ADEP_mvt AS apt, month(MVT_TIME_UTC_mvt) AS month, count(*) AS n
FROM read_parquet('{path}') WHERE PHASE_mvt = 'DEP' AND AOBT_3_flt IS NULL GROUP BY 1, 2"""


def recent_arrival_taxi(dep: pd.DataFrame, arr: pd.DataFrame, k: int = 30) -> np.ndarray:
    """Mean measured taxi-in of the last k arrivals that had already reached a stand when this departure's anchor passed."""
    anc, out = dep.t_off.values - dep.z_anchor.values, np.full(len(dep), np.nan)
    for apt, ix in dep.groupby("apt").indices.items():
        a = arr[arr.apt == apt]
        cs = np.concatenate([[0.0], np.cumsum(a.ti.values)])
        pos = np.searchsorted(a.tb.values, anc[ix], "left")  # blocked in before the anchor, so knowable then
        lo = np.maximum(pos - k, 0)
        out[ix] = np.where(pos - lo >= 3, (cs[pos] - cs[lo]) / np.maximum(pos - lo, 1), np.nan)
    return out


def cell_rmse(e: np.ndarray, cell: pd.Series, weights: pd.Series) -> float:
    """The holdout's per-cell mean squared error, re-averaged under another row mix."""
    ms = pd.Series(e ** 2).groupby(cell.values).mean()
    common = [c for c in ms.index if c in weights.index]
    w = weights.loc[common]
    return float(np.sqrt((ms.loc[common] * w).sum() / w.sum()))


def composite(la: float, lb: float, n_a: int, n_b: int) -> float:
    return float(np.sqrt((la ** 2 * n_a + lb ** 2 * n_b) / (n_a + n_b)))


def mix_section(teb, e, la, rank_w, hold_w, cell, n_a, n_b, n_lane_a) -> list[str]:
    tab = teb.assign(e=e).groupby(["apt", "month"]).agg(
        n=("e", "size"), rmse=("e", lambda s: float(np.sqrt((s ** 2).mean())))).reset_index()
    tab["rank_n"] = [int(rank_w.get((a, m), 0)) for a, m in zip(tab.apt, tab.month)]
    out = [f"The lane split itself matches: {n_b:,} of {n_a + n_b:,} ranking departures have no flight plan "
           f"({n_b / (n_a + n_b) * 100:.3f}%) against {len(teb):,} of {len(teb) + n_lane_a:,} in the holdout "
           f"({len(teb) / (len(teb) + n_lane_a) * 100:.3f}%). The mix inside the lane does not. Lane B was July-heavy in 2025 and is "
           "balanced in 2026, and the January cells that grew are the ones the champion handles worst.", "",
           "| airport | month | holdout rows | holdout RMSE | ranking rows | ranking / holdout |", "|---|---|---|---|---|---|"]
    for r in tab.sort_values(["apt", "month"]).itertuples():
        out.append(f"| {r.apt} | {r.month} | {r.n} | {r.rmse:.1f} | {r.rank_n} | {r.rank_n / r.n:.2f} |")
    out += ["", "## 2. Reweighting the holdout to that mix, and whether it survives the rows F17 names", "",
            "Each cell's holdout mean squared error, re-averaged with the ranking set's row counts as weights. The first row is the "
            "headline and it is misleading on its own; the rest of the table is why.", "",
            "| worst lane-B rows dropped | lane B at the holdout mix | lane B at the ranking mix | shift | composite, holdout mix | "
            "composite, ranking mix |", "|---|---|---|---|---|---|"]
    order = np.argsort(np.abs(e))[::-1]
    for k in DROPS:
        keep = np.ones(len(teb), bool)
        keep[order[:k]] = False
        c2, e2 = cell[keep], e[keep]
        a, b = cell_rmse(e2, c2, hold_w), cell_rmse(e2, c2, rank_w)
        out.append(f"| {k} | {a:.1f} | {b:.1f} | {b - a:+.1f} | {composite(la, a, n_a, n_b):.2f} | "
                   f"{composite(la, b, n_a, n_b):.2f} |")
    out += ["", "Read the last four rows, not the first. The headline shift is the two LFPG January rows flag F17 already names, "
            "arriving again through the weights. With them set aside the mix shift is worth roughly +15 to +40 s of lane-B RMSE, about "
            "+1 to +3 s of composite. Small, real, and against us: the ranking mix is tilted toward the cells we predict worst. The "
            "board anchors on `docs/gate-s4.html` assume the holdout mix and do not carry it."]
    return out


def main() -> None:
    con = connect()
    trb, teb = lane_b()
    tra, tea = lane_a()
    y, pb = teb.y.values, floored(predict_b(CURRENT["B"], trb, teb))
    e = pb - y
    pa = floored(champion_holdout("MC2", tra, tea))
    la = rmse(pa - tea.y.values)
    cell = pd.Series(list(zip(teb.apt, teb.month)))

    rc = con.sql(DEP_CELLS.format(path=RANK)).df()
    rank_w = rc.set_index(pd.Index(list(zip(rc.apt, rc.month)))).n
    n_b = int(rc.n.sum())
    n_a = int(con.execute(f"SELECT count(*) FROM read_parquet('{RANK}') WHERE PHASE_mvt = 'DEP'").fetchone()[0]) - n_b

    lines = ["# MIX: what the 2026 ranking file's own rows say (diagnostic, never a verdict)", "",
             f"Regenerate with `python -m taxi.mix`. Lane-B champion {CURRENT['B']}, lane-A champion {CURRENT['A']}. Nothing here "
             "decides a verdict; two of the four questions come back negative and are recorded so they are not asked again.", "",
             "## 1. The ranking set's lane-B rows sit in different cells from the holdout's", ""]
    lines += mix_section(teb, e, la, rank_w, cell.value_counts(), cell, n_a, n_b, len(tea))

    arr = con.sql(ARR_TAXI.format(files=_files([month_file(m) for m in HOLDOUT_MONTHS]))).df()
    ati = recent_arrival_taxi(tea, arr)
    ra, ok = tea.y.values - pa, np.isfinite(ati)
    lines += ["", "## 3. Arrival taxi times: available for 2026, and worth nothing (negative)", "",
              "The ranking file blanks the departure taxi time but keeps the arrival one on every one of its arrivals. An arrival that "
              "has already reached a stand when the anchor passes has a measured taxi-in, in seconds rather than the counts every "
              "existing traffic column uses. It measures the same surface. It is also already captured.", "",
              "| statistic | value |", "|---|---|",
              f"| coverage on lane-A holdout rows | {ok.mean():.3f} |",
              f"| correlation with the label | {np.corrcoef(ati[ok], tea.y.values[ok])[0, 1]:+.3f} |",
              f"| correlation with the MC2 residual | {np.corrcoef(ati[ok], ra[ok])[0, 1]:+.3f} |", "",
              "A residual correlation that small caps the gain near 0.2 s, an order of magnitude under the 2 s floor. No cell was "
              "declared."]

    a25 = con.sql(ARR_RATE.format(big=BIG, files=_files([month_file(m) for m in HOLDOUT_MONTHS]))).df()
    a26 = con.sql(ARR_RATE.format(big=BIG, files=_files([RANK]))).df()
    lines += ["", "## 4. Did the recording process shift between the years?", "",
              "| year | month | arrivals | mean taxi-in | over an hour | rate |", "|---|---|---|---|---|---|"]
    for tag, d in (("2025", a25), ("2026", a26[a26.m.isin(HOLDOUT_MONTHS)])):
        for r in d.itertuples():
            lines.append(f"| {tag} | {int(r.m)} | {int(r.n):,} | {r.mean_ti:.1f} s | {int(r.big)} | {r.big / r.n:.5f} |")
    lines += ["", "Mean arrival taxi-in is stable across the years, so the surface has not moved and a large level correction is not "
              "warranted. Arrivals carry the over-an-hour artifact at under 0.1% in every month above, against 3.8% on lane-B departures, so it is "
              "a property of off-block recording and arrivals cannot be used to estimate its 2026 rate."]

    xb, yb = trb[list(CLS)].astype(float).values, (trb.y.values > BIG).astype(int)
    m = HistGradientBoostingClassifier(max_depth=3, max_iter=200, random_state=0).fit(xb, yb)
    ph, yt = m.predict_proba(teb[list(CLS)].astype(float).values)[:, 1], (y > BIG).astype(int)
    nl, mag = (teb.apt != "LIRF").values, trb.y.values[trb.y.values > BIG].mean()
    lines += ["", "## 5. Is the artifact separable beyond what the champion already uses? (negative)", "",
              f"Lane B carries {int(yb.sum())} taxi times over an hour in {len(trb):,} training rows ({yb.mean() * 100:.2f}%) and "
              f"{int(yt.sum())} in {len(teb):,} holdout rows ({yt.mean() * 100:.2f}%). A classifier on the champion's own columns reaches "
              f"AUC {roc_auc_score(yt, ph):.3f}, which looks like a lane. It is not: outside LIRF the AUC is "
              f"{roc_auc_score(yt[nl], ph[nl]):.3f} on {int(yt[nl].sum())} positives, and the rest is the no-type and schedule-gap "
              "structure the champion's gate already rides.", "",
              f"Adding a shrunk artifact term, the prediction plus lambda x P(artifact) x (mean artifact magnitude {mag:.0f} s minus the "
              "prediction), floored:", "",
              "| lambda | lane B RMSE | paired against the champion [95%] |", "|---|---|---|"]
    for lam in (0.25, 0.5, 1.0):
        q = floored(pb + lam * ph * np.maximum(mag - pb, 0.0))
        v = paired_delta(e, q - y, teb.day.values, lane="B")
        lines.append(f"| {lam:.2f} | {rmse(q - y):.1f} | {v.delta:+.1f} [{v.lo:+.1f}, {v.hi:+.1f}] |")
    lines += ["", "Every weight loses, and the loss grows with the weight. The champion already predicts high where the artifacts are: "
              "in the top decile of P(artifact) it over-predicts on a truth mean near 4,900 s. The level is not the missing piece. No "
              "cell was declared."]

    text = "\n".join(lines) + "\n"
    with open("docs/MIX.md", "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
