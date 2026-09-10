"""LP1: TabPFN on lane B, the lane that carries half the composite squared error and whose champion reads one column.

L8c predicts from z_sched plus an airline-by-band copy probability; L1's tree cells saw about ten columns and lost. Fourteen columns that describe
where the aircraft is standing and how busy the airport is have never been shown to a lane-B model. TabPFN is pretrained for exactly this size, so
this fits all 17,097 training rows at once on raw y and lets the model learn the copy mixture L6 hand-codes.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
from tabpfn import TabPFNRegressor

from taxi.cb1 import lane_b
from taxi.champ import CURRENT, predict_b
from taxi.data import CV_FOLDS
from taxi.l2 import floored
from taxi.verdict import paired_delta, rmse
from taxi.w1 import verdict_line

CHUNK = 4_096
COLS = ["apt", "hour", "wday", "z_sched", "z_anchor", "runway", "stand", "actype", "queue", "dep_before", "dep_after",
        "arr_anchor", "arr_off", "push_anchor", "frule", "notyp", "occ", "airline", "rwy_share", "hol", "hol_prev", "hol_next", "rq", "rec_taxi"]


def numeric(fit: pd.DataFrame, va: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Categoricals as integer codes shared by both frames, so a level means the same thing on each side."""
    both = pd.concat([fit[COLS], va[COLS]]).reset_index(drop=True)
    cats = [c for c in COLS if str(both[c].dtype) in ("category", "object")]
    for c in cats:
        codes = pd.Categorical(both[c]).codes.astype(float)
        both[c] = np.where(codes < 0, np.nan, codes)
    x = both.values.astype(float)
    return x[: len(fit)], x[len(fit):], [COLS.index(c) for c in cats]


def fit_predict(fit: pd.DataFrame, va: pd.DataFrame) -> tuple[np.ndarray, float, float]:
    x_fit, x_va, cats = numeric(fit, va)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    reg = TabPFNRegressor(device="cuda", categorical_features_indices=cats, random_state=0)
    reg.fit(x_fit, fit.y.values)
    p = np.concatenate([np.asarray(reg.predict(x_va[i:i + CHUNK])) for i in range(0, len(x_va), CHUNK)])
    dt, vram = time.time() - t0, torch.cuda.max_memory_allocated() / 1e9
    del reg
    torch.cuda.empty_cache()
    return p, dt, vram


def main() -> None:
    trb, teb = lane_b()
    print(f"lane B: {len(trb):,} training rows, {len(teb):,} holdout rows, {len(COLS)} columns", flush=True)

    lines = ["| fold | rows | L8c RMSE (s) | LP1 RMSE (s) | fit s | VRAM GB |", "|---|---|---|---|---|---|"]
    for fold in CV_FOLDS:
        m = trb.month.isin(fold).values
        p, dt, vram = fit_predict(trb[~m], trb[m])
        y = trb.y.values[m]
        base = floored(predict_b(CURRENT["B"], trb[~m], trb[m]))
        lines.append(f"| {fold} | {int(m.sum()):,} | {rmse(base - y):.1f} | {rmse(floored(p) - y):.1f} | {dt:.0f} | {vram:.1f} |")
        print(lines[-1], flush=True)

    p, dt, vram = fit_predict(trb, teb)
    lp1, base, y = floored(p), floored(predict_b(CURRENT["B"], trb, teb)), teb.y.values
    v = paired_delta(base - y, lp1 - y, teb.day.values, lane="B")
    jan, jul = teb.month.values == 1, teb.month.values == 7
    row = lambda n, e: f"| {n} | {rmse(e):.1f} | {rmse(e[jan]):.1f} | {rmse(e[jul]):.1f} |"  # noqa: E731

    out = ["# LP1: TabPFN on lane B, Jan + Jul 2025 development holdout", "",
           f"Regenerate with `python -m taxi.lp1`. All {len(trb):,} training rows in one fit on raw y, {len(COLS)} columns, predicted in "
           f"{CHUNK:,}-row chunks and floored at 0 s. Built with PriorLabs-TabPFN (Apache 2.0 plus attribution). Holdout fit {dt:.0f} s, "
           f"peak VRAM {vram:.1f} GB.", "", "## Month-pair folds", ""] + lines
    out += ["", f"## Holdout verdict vs champion {CURRENT['B']} ({len(teb):,} rows)", "",
            "| model | RMSE | Jan | Jul |", "|---|---|---|---|", row(CURRENT["B"], base - y), row("LP1", lp1 - y), "",
            verdict_line("LP1", CURRENT["B"], v, k=1)]
    text = "\n".join(out) + "\n"
    with open("docs/LP1.md", "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
