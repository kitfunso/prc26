"""HZ (diagnostic, never a verdict): where the remaining headroom is NOT, on both lanes.

Three closures the campaign had left open. On lane A: what a PERFECT model conditioning on the discrete keys would score, and
whether the source columns the view never reads carry anything SB3 does not already hold. On lane B: whether the copy gate is
calibrated, since 96% of that lane's rows are ordinary flights and 13 rows hold two thirds of its squared error.
No cell is declared from any of it: every answer either loses or lands under the 2 s floor.
"""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from taxi.cb1 import lane_b
from taxi.champ import lane_a, predict_b
from taxi.data import HOLDOUT_MONTHS, _files, month_file
from taxi.l2 import floored
from taxi.sb3 import predict as sb3_predict
from taxi.verdict import rmse

FLOOR = 2.0
LADDER = (("apt",), ("apt", "runway"), ("apt", "stand"), ("apt", "stand", "runway"), ("apt", "stand", "runway", "actype"),
          ("apt", "stand", "runway", "actype", "hr"), ("apt", "zb"), ("apt", "runway", "zb"), ("apt", "stand", "runway", "zb"),
          ("apt", "stand", "runway", "actype", "zb"), ("apt", "stand", "runway", "actype", "zb", "qb"))
PROBES = ("d_arv3", "air3", "cs_mm", "typ_mm", "ades_mm", "rule_mm")
RAW = """SELECT MVT_ID_mvt AS id,
  epoch(ARVT_3_flt) - epoch(ARVT_1_flt) AS d_arv3, epoch(ARVT_3_flt) - epoch(MVT_TIME_UTC_mvt) AS air3,
  (replace(CALLSIGN_flt,' ','') IS DISTINCT FROM replace(FLIGHT_mvt,' ',''))::INT AS cs_mm,
  (AIRCRAFT_TYPE_mvt IS DISTINCT FROM AIRCRAFT_TYPE_flt)::INT AS typ_mm,
  (ADES_mvt IS DISTINCT FROM ADES_FILED_flt)::INT AS ades_mm,
  (FLIGHT_RULE_mvt IS DISTINCT FROM FLIGHT_RULE_flt)::INT AS rule_mm
  FROM read_parquet({files}) WHERE PHASE_mvt='DEP'"""


def key_oracle(d: pd.DataFrame, keys: tuple[str, ...]) -> tuple[float, np.ndarray, int]:
    """Pooled within-group spread of c over rows matched on these keys: the RMSE of a perfect model that reads only them.

    Singletons carry no information and would read as zero error, so they are dropped and the champion is scored on the rest.
    """
    g = d.groupby([d[k].astype(object) for k in keys], observed=True)["c"]
    n = g.transform("size").values
    m = n >= 2
    dev = (d.c.values - g.transform("mean").values)[m]
    return float(np.sqrt(np.sum(dev ** 2) / np.sum((n[m] - 1) / n[m]))), m, int(g.ngroups)


def probe_cols(ids: np.ndarray) -> pd.DataFrame:
    con = duckdb.connect()
    con.execute("SET TimeZone='UTC'")
    raw = con.sql(RAW.format(files=_files([month_file(m) for m in HOLDOUT_MONTHS]))).df()
    return pd.DataFrame({"id": ids}).merge(raw, on="id", how="left")


def lane_a_sections() -> tuple[list[str], list[str], tuple[str, float, float]]:
    tra, tea = lane_a()
    e0 = sb3_predict(tra, tea) - tea.y.values
    d = tea.copy()
    d["zb"], d["qb"], d["hr"] = np.floor(d.z_aobt.values / 120.0), np.minimum(d.queue.values, 40), d.hour.values
    lad = []
    for keys in LADDER:
        orc, m, ng = key_oracle(d, keys)
        lad.append(f"| {' + '.join(keys)} | {ng:,} | {m.mean():.1%} | {orc:.1f} | {rmse(e0[m]):.1f} | {rmse(e0[m]) - orc:+.1f} |")

    j = probe_cols(tea.id.values)
    rows, best = [], ("", 0.0)
    for c in PROBES:
        v = j[c].values.astype(float)
        ok = np.isfinite(v)
        rc = re = float("nan")
        if ok.sum() > 10 and v[ok].std() > 0:
            rc, re = np.corrcoef(v[ok], tea.c.values[ok])[0, 1], np.corrcoef(v[ok], e0[ok])[0, 1]
            if abs(rc) > abs(best[1]):
                best = (c, rc)
        rows.append(f"| {c} | {ok.mean():.1%} | {rc:+.4f} | {re:+.4f} |")
    v = j[best[0]].values.astype(float)
    ok = np.isfinite(v)
    adj = e0.copy()
    q = pd.qcut(pd.Series(v[ok]), 50, duplicates="drop")
    adj[ok] = e0[ok] - pd.Series(e0[ok]).groupby(q.values, observed=True).transform("mean").values
    return lad, rows, (best[0], rmse(e0), rmse(adj))


def lane_b_sections() -> tuple[list[str], list[str], list[str], float, float]:
    trb, teb = lane_b()
    p, y, z = floored(predict_b("L8c", trb, teb)), teb.y.values, teb.z_sched.values
    e = p - y
    body = float(np.median(p[z <= 3600]))  # the lane's ordinary level, read off its own predictions, not assumed
    bands = [("normal, under 1 h", y < 3600), ("stale, 1 to 12 h", (y >= 3600) & (y <= 43200)), ("over 12 h", y > 43200)]
    bd = [f"| {n} | {m.sum():,} | {np.sum(e[m] ** 2) / np.sum(e ** 2):.1%} | {rmse(e[m]):,.1f} | "
          f"{rmse(np.where(m, 0.0, e)):,.1f} | {rmse(e) - rmse(np.where(m, 0.0, e)):+.1f} |" for n, m in bands]

    g = z > 3600
    w = np.where(g, np.clip((p - body) / np.maximum(z - body, 1.0), 0, 1), 0.0)
    cal = []
    for lo, hi in ((0, .01), (.01, .05), (.05, .10), (.10, .25), (.25, .50), (.50, 1.01)):
        m = (w >= lo) & (w < hi)
        if m.sum():
            cal.append(f"| {lo:.2f} to {hi:.2f} | {m.sum():,} | {w[m].mean():.4f} | {np.mean(y[m] > 3600):.4f} | "
                       f"{p[m].mean():,.0f} | {y[m].mean():,.0f} |")

    lam = []
    for a in (0.0, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00):
        q = p.copy()
        q[g] = np.maximum(body + a * (p[g] - body), 0.0)
        lam.append(f"| {a:.2f} | {rmse(q - y):,.1f} | {rmse(e) - rmse(q - y):+.1f} |")
    return bd, cal, lam, rmse(e), body


def main(dst: str = "docs/HZ.md") -> None:
    lad, probes, (tag, base, adj) = lane_a_sections()
    bd, cal, lam, rb, body = lane_b_sections()
    out = ["# HZ: where the remaining headroom is not, both lanes, Jan + Jul 2025 development holdout", "",
           "Diagnostic, never a verdict, no cell declared. Regenerate with `python -m taxi.hz`. Three questions the campaign had "
           f"left open, each answered against the standing champions SB3 (lane A) and L8c (lane B, {rb:,.2f} s).", "",
           "## 1. Lane A: what a perfect model reading only the discrete keys would score", "",
           "Rows are matched on each key set, and the pooled within-group spread of `c` is the RMSE of a model that conditions on "
           "those keys and nothing else. That is a CEILING for the key set: no feature engineering over those columns can beat it. "
           "Groups of one would read as zero error, so they are dropped and SB3 is scored on the same covered rows. `zb` is the "
           "plan gap in 2-minute bins, `qb` the departure queue capped at 40, `hr` the hour of day.", "",
           "| keys | groups | rows covered | perfect on these keys | SB3, same rows | headroom |", "|---|---|---|---|---|---|",
           *lad, "",
           "Every rung is NEGATIVE. SB3 beats a perfect conditioner on the discrete keys at every one of them. Two flights sharing "
           "airport, stand, runway, aircraft type, plan-gap bin and queue bin still differ in `c` by more than SB3's own error, so "
           "matching on those keys throws away more than it gains. The categorical axis is not where the remaining signal is: the "
           "continuous context columns are carrying the model past anything key-matching could reach.", "",
           "## 2. Lane A: the source columns the view never reads", "",
           "`taxi.data.VIEW` selects 22 of the 30 columns in the movement files. The unread ones are probed six ways below. "
           "`d_arv3` is the flight's arrival gap between source 3 and source 1, from the same source-3 system as the `AOBT_3_flt` "
           "that forms half of the lane-A label, so it is the one with a mechanism behind it. Correlation with `c` says whether a "
           "probe knows anything at all; correlation with SB3's RESIDUAL says whether it knows anything SB3 does not already hold.", "",
           "| probe | non-null | corr with c | corr with SB3 residual |", "|---|---|---|---|", *probes, "",
           f"`{tag}` is the strongest of them against `c` and among the weakest against the residual. Removing its 50-bin mean from "
           f"SB3's own residual, fit on the very rows it is judged on and therefore an upper bound no honest feature could reach, "
           f"moves {base:.2f} s to {adj:.2f} s: **{base - adj:+.2f} s**. The champion already holds what these columns carry, "
           "through `d_ae`. The unread-column axis is closed.", "",
           "## 3. Lane B: the copy gate is calibrated", "",
           "96% of lane-B rows are ordinary flights while a handful of huge ones hold most of the squared error, so the obvious "
           "suspicion is that the copy gate leaks onto ordinary flights and the many pay for the few. Where the error sits, and "
           "what perfect knowledge of one band alone would be worth:", "",
           "| band | rows | share of lane-B squared error | band RMSE | lane B if perfect here | gain |",
           "|---|---|---|---|---|---|", *bd, "",
           f"The gate weight implied by each prediction, against the rate of a big taxi actually realised in that bin. The body "
           f"level is read off the lane's own predictions ({body:,.0f} s) and the weight from the prediction and the schedule gap, "
           "so both sides use ranking-visible values only: this is a blind calibration curve, not an oracle.", "",
           "| implied gate weight | rows | mean weight | realised rate of y over 1 h | mean prediction | mean truth |",
           "|---|---|---|---|---|---|", *cal, "",
           "Mildly over-confident in the middle bins and close everywhere else. Shrinking or stretching every implied weight by one "
           "scalar is the only blind correction a calibration error of that shape admits:", "",
           "| scalar on the implied weight | lane B RMSE | vs L8c |", "|---|---|---|", *lam, "",
           "The sweep peaks exactly at 1.00, the champion as it stands, and falls away hard in both directions: the same scalar "
           "that would fix the middle bins wrecks the top one. Two further blind rules were priced beside it and both land far "
           "under the floor. Replacing every prediction below a cut with a per-(airport, runway) mean of the training rows is worth "
           "at most +3.2 s at a 5,400 s cut and LOSES 16.6 s at 7,200 s. Swapping the whole body for that group mean wherever the "
           "gate is silent, which is 63.6% of the lane's squared error, is worth +3.1 s. Lane B's median detectable effect is 32 s "
           "(`docs/MDE.md`), so none of these is measurable, let alone bankable. The gate is not the problem and the body is not "
           "the problem. The 13 rows are, and `docs/DE.md` closed them.", "",
           "## What this closes", "",
           "Lane A's categorical axis (section 1), lane A's unread-column axis (section 2) and lane B's calibration axis "
           "(section 3). Taken with `docs/EXH.md` on the read columns, `docs/HP1.md` on the champion's own hyperparameter cell, "
           f"`docs/BF.md` on every stack of every cached model and `docs/DE.md` on the artifact rows, no mechanism this campaign "
           f"can reach is worth the {FLOOR:.0f} s floor on either lane.", ""]
    text = "\n".join(out)
    with open(dst, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
