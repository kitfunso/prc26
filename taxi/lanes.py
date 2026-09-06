"""Single-change lanes on MC2's cell (WS1, T1, G1, TW1, R1, H1): month-pair CV picks among a lane's cells, the CV winner alone meets the holdout."""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from taxi.champ import COLS_D1, _fingerprint, champion_holdout, champion_oof, lane_a
from taxi.data import WINDOW
from taxi.m1 import final_rounds
from taxi.m2 import is_tail
from taxi.mc1 import CV, Prep, cv_a, predict_cb, stages_cb
from taxi.verdict import paired_delta, rmse
from taxi.w1 import verdict_line

CACHE = "data/cache/lanes"
MIN_ROWS = 20  # G1: a target-encoding group needs this many fit rows, else the (airport, stand) fallback, else NaN


@dataclass(frozen=True)
class Lane:
    cols: tuple[str, ...] = tuple(COLS_D1)
    window: float = WINDOW
    lag: float = 0.0
    prep: Prep | None = None
    inv: Callable | None = None
    fit_kw: dict = field(default_factory=dict)
    rounds: tuple[int, int, int] | None = None  # set = no CV, fit at these rounds (WS1 keeps MC2's recorded trees: it is a stress, not a search)


def _median_by(fit: pd.DataFrame, rows: pd.DataFrame, keys: list[str]) -> np.ndarray:
    g = fit.groupby([fit[k].astype(object) for k in keys]).c.agg(["median", "size"])
    ok = g[g["size"] >= MIN_ROWS]["median"]
    return ok.reindex(pd.MultiIndex.from_arrays([rows[k].astype(object).values for k in keys])).values


def _encode(fit: pd.DataFrame, rows: pd.DataFrame) -> np.ndarray:
    v = _median_by(fit, rows, ["apt", "stand", "runway"])
    return np.where(np.isnan(v), _median_by(fit, rows, ["apt", "stand"]), v)


def target_encode(fit: pd.DataFrame, rows: pd.DataFrame | None) -> pd.DataFrame:
    """G1: median of c per (airport, stand, runway) over the fit rows; the fit rows themselves are encoded from the other fit months."""
    if rows is not None:
        return rows.assign(te_stand=_encode(fit, rows))
    out = np.full(len(fit), np.nan)
    for m in fit.month.unique():
        k = (fit.month == m).values
        out[k] = _encode(fit[~k], fit[k])
    return fit.assign(te_stand=out)


MC2_ROUNDS = lambda: final_rounds("MC2", "docs/MC2.md")  # noqa: E731  read at run time: docs/MC2.md is the recorded source
LANES: dict[str, tuple[Callable[[], dict[str, Lane]], int, str, bool]] = {
    "WS1": (lambda: {"WS1": Lane(lag=600.0, rounds=MC2_ROUNDS())}, 1,
            "Weather join-shift stress: the as-of join reads the latest observation at or before the anchor minus 600 s; MC2's cell at its recorded "
            "rounds on the shifted columns, no CV. A loss says the join is fragile to observation latency, a tie says it is not.", False),
    "T1": (lambda: {"T1a": Lane(fit_kw=dict(fwd=np.arcsinh), inv=np.sinh), "T1b": Lane(fit_kw=dict(loss="MAE"))}, 2,
           "Lane-A target treatment on MC2's cell: T1a's regressors fit asinh(c) and predict sinh of the output; T1b's regressors fit c with the MAE "
           "loss (a median); the tail classifier is unchanged in both.", False),
    "G1": (lambda: {"G1": Lane(cols=(*COLS_D1, "te_stand"), prep=target_encode)}, 1,
           f"Fold-internal target encoding: the median of c per (airport, stand, runway) over the fit rows with at least {MIN_ROWS} rows, else per "
           "(airport, stand), else NaN, as one numeric column on MC2's cell; fit rows are encoded from the other fit months, fold and holdout rows "
           "from all fit rows.", False),
    "TW1": (lambda: {f"TW{w}": Lane(window=float(w)) for w in (300, 1800, 3600)}, 4,
            "Traffic window sweep: the six counts rebuilt at 300, 1,800 and 3,600 s (MC2 uses 900 s) on MC2's cell; the CV winner meets the holdout "
            "only if its CV beats MC2's own CV on the same folds and data.", True),
    "R1": (lambda: {"R1": Lane(cols=(*COLS_D1, "rwy_share"))}, 1,
           "Runway-configuration proxy: the share of the airport's takeoffs on this departure's runway in the hour before the anchor (NaN when that "
           "hour is empty), one column on MC2's cell.", False),
    "TH1": (lambda: {f"TH{v}": Lane(fit_kw=dict(tail=float(v))) for v in (400, 900, 1200)}, 4,
            "Tail split sweep: the |c| threshold that splits MC2's two regressors, at 400, 900 and 1,200 s (MC2 uses 600) on MC2's cell; the "
            "classifier target moves with it. The CV winner meets the holdout only if its CV beats MC2's own CV on the same folds and data.", True),
    "H1": (lambda: {"H1": Lane(cols=(*COLS_D1, "hol", "hol_prev", "hol_next"))}, 1,
           "Public holiday in the airport's country on the takeoff day, the day before and the day after (`holidays` 0.103, country from the ICAO "
           "prefix), three flags on MC2's cell.", False),
}


def cv_cell(lane_id: str, name: str, c: Lane, tra: pd.DataFrame, champ: str) -> CV | None:
    """The cell's CV, cached so a killed queue resumes at the next cell; None when the cell declares its rounds."""
    if c.rounds is not None:
        return None
    path, fp = f"{CACHE}/{lane_id}_{name}", _fingerprint(tra, tra.iloc[:0])
    if os.path.exists(path + ".json"):
        with open(path + ".json", encoding="utf-8") as fh:
            j = json.load(fh)
        if j["fp"] == fp:
            return CV(tuple(j["rounds"]), j["lines"], pd.read_parquet(path + ".parquet").oof.values, j["rmse"])
    r = cv_a(tra, list(c.cols), name, prep=c.prep, inv=c.inv, ref=champ, **c.fit_kw)
    os.makedirs(CACHE, exist_ok=True)
    pd.DataFrame({"id": tra.id.values, "oof": r.oof}).to_parquet(path + ".parquet", index=False)
    with open(path + ".json", "w", encoding="utf-8") as fh:
        json.dump(dict(fp=fp, rounds=list(r.rounds), rmse=r.rmse, lines=r.lines), fh)
    return r


def holdout(c: Lane, rounds: tuple[int, int, int], tra: pd.DataFrame, tea: pd.DataFrame) -> np.ndarray:
    f, t = (c.prep(tra, None), c.prep(tra, tea)) if c.prep else (tra, tea)
    return predict_cb(stages_cb(f, iters=rounds, cols=list(c.cols), **c.fit_kw), t, list(c.cols), c.inv)[0]


def run(lane_id: str, champ: str = "MC2") -> None:
    make, k, blurb, needs_cv_win = LANES[lane_id]
    cells, frames = make(), {}

    def get(c: Lane) -> tuple[pd.DataFrame, pd.DataFrame]:
        key = (c.window, c.lag)
        if key not in frames:
            frames[key] = lane_a(*key)
        return frames[key]

    tra0, tea0 = get(Lane())
    y, days = tea0.y.values, tea0.day.values
    e0 = champion_holdout(champ, tra0, tea0) - y
    ref_cv = rmse(champion_oof(champ, tra0) - tra0.y.values)
    results: dict[str, CV | None] = {}
    for name, c in cells.items():
        tra, tea = get(c)
        assert np.array_equal(tra.id.values, tra0.id.values) and np.array_equal(tea.id.values, tea0.id.values)
        results[name] = cv_cell(lane_id, name, c, tra, champ)
    cvd = {n: r.rmse for n, r in results.items() if r is not None}
    best = min(cvd, key=cvd.get) if cvd else next(iter(cells))
    lines = [f"# {lane_id}: {blurb.split(':')[0]}, Jan+Jul 2025 development holdout", "",
             f"Regenerate with `python -m taxi.lanes {lane_id}`. {blurb} Cells: {', '.join(cells)}. k = {k}; the CV winner alone meets the holdout "
             f"against {champ}'s cached holdout predictions. {champ} on the same folds and data (`taxi.champ.champion_oof`): CV {ref_cv:.1f} s.", ""]
    for name, r in results.items():
        lines += [f"## {name}: month-pair CV", ""] + (r.lines if r else [f"No CV: fit at {champ}'s recorded rounds {cells[name].rounds}."]) + [""]
    if cvd:
        lines += ["CV RMSE by cell: " + ", ".join(f"{n} {v:.1f} s" for n, v in cvd.items()) + f"; {champ} {ref_cv:.1f} s; CV winner {best}.", ""]
    if needs_cv_win and cvd[best] >= ref_cv:
        lines += [f"## Holdout: not touched", "", f"{best} ({cvd[best]:.1f} s) does not beat {champ} on the same folds ({ref_cv:.1f} s); by the "
                  f"declaration no cell meets the holdout and {champ} stays."]
    else:
        c, tra, tea = cells[best], *get(cells[best])
        rounds = c.rounds or results[best].rounds
        path = f"{CACHE}/{lane_id}_{best}_holdout.parquet"
        if os.path.exists(path) and np.array_equal(pd.read_parquet(path).id.values, tea.id.values):
            p1 = pd.read_parquet(path).pred.values
        else:
            p1 = holdout(c, rounds, tra, tea)
            os.makedirs(CACHE, exist_ok=True)
            pd.DataFrame({"id": tea.id.values, "pred": p1}).to_parquet(path, index=False)
        e1 = p1 - y
        v = paired_delta(e0, e1, days)
        vt, jan, jul = is_tail(tea0), tea0.month.values == 1, tea0.month.values == 7
        row = lambda n, e: f"| {n} | {rmse(e):.1f} | {rmse(e[jan]):.1f} | {rmse(e[jul]):.1f} | {rmse(e[vt]):.1f} | {rmse(e[~vt]):.1f} |"  # noqa: E731
        lines += [f"## Holdout verdict vs champion {champ} ({len(tea0)} rows; tail rows {int(vt.sum())}; {best} fit at rounds {rounds})", "",
                  "| model | RMSE | Jan | Jul | tail rows | body rows |", "|---|---|---|---|---|---|", row(champ, e0), row(best, e1), "",
                  verdict_line(best, champ, v, k=k)]
    text = "\n".join(lines) + "\n"
    with open(f"docs/{lane_id}.md", "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    run(sys.argv[1])
