"""M3 pilot (protocol section 4, R12): TabPFN regressor on one airport's lane-A rows, month-pair OOF against the champion M1, three gates."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tabpfn import TabPFNRegressor

from taxi.data import CATEGORICAL, CV_FOLDS, TRAIN_MONTHS, load
from taxi.m1 import MODEL_FEATURES, chosen_cells, predict, train
from taxi.verdict import FLOOR, rmse

AIRPORT = "LIRF"  # largest lane-A SSE share on the holdout (docs/M1.md, 23.0%)
LIMIT = 50_000  # TabPFN 6.3.1 pretraining limit on rows; a seeded subsample of the fold's rows
CHUNK = 4_096  # predict in slices: one 23k-row predict held 11 GB of RAM and the harness killed the run
COLS = [c for c in MODEL_FEATURES if c != "apt"]
GATES = dict(minutes=30.0, vram_gb=16.0, gain_s=FLOOR)
CACHE = Path("data/m3_folds.json")  # one entry per finished fold: the harness kills long jobs when the box runs low on RAM


def numeric(fit: pd.DataFrame, va: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Categoricals as integer codes shared by both frames (missing stays NaN); TabPFN takes those columns by index."""
    both = pd.concat([fit[COLS], va[COLS]]).reset_index(drop=True)
    for c in CATEGORICAL:
        codes = pd.Categorical(both[c]).codes.astype(float)
        both[c] = np.where(codes < 0, np.nan, codes)
    x = both.values.astype(float)
    return x[: len(fit)], x[len(fit):], [COLS.index(c) for c in CATEGORICAL]


def main() -> None:
    tr = load(TRAIN_MONTHS)
    tra = tr[tr.has_fpl].reset_index(drop=True)
    cell, _ = chosen_cells()["A"]
    lines = ["| fold | fit rows (of the airport's rows outside the fold) | predicted rows | M1 RMSE (s) | M3 RMSE (s) | fit + predict s | peak VRAM GB |",
             "|---|---|---|---|---|---|---|"]
    sq1 = sq3 = secs = peak = 0.0
    n = 0
    apt = (tra.apt == AIRPORT).values
    done = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    for fold in CV_FOLDS:
        if str(fold) not in done:
            m = tra.month.isin(fold).values
            fit, va = tra[~m & apt], tra[m & apt]
            sub = fit.sample(n=min(LIMIT, len(fit)), random_state=0)
            x_fit, x_va, cats = numeric(sub, va)
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            reg = TabPFNRegressor(device="cuda", categorical_features_indices=cats, random_state=0)
            reg.fit(x_fit, sub.c.values)
            p3 = np.concatenate([np.asarray(reg.predict(x_va[i:i + CHUNK])) for i in range(0, len(x_va), CHUNK)]) + va.z_aobt.values
            dt, vram = time.time() - t0, torch.cuda.max_memory_allocated() / 1e9
            e1 = predict(train(tra[~m], "A", cell, va=tra[m]), va, "A") - va.y.values
            e3 = p3 - va.y.values
            done[str(fold)] = dict(n_sub=len(sub), n_fit=len(fit), n_va=len(va), sq1=float(e1 @ e1), sq3=float(e3 @ e3), dt=dt, vram=vram)
            CACHE.write_text(json.dumps(done, indent=1))
        r = done[str(fold)]
        sq1, sq3, n, secs, peak = sq1 + r["sq1"], sq3 + r["sq3"], n + r["n_va"], secs + r["dt"], max(peak, r["vram"])
        lines.append(f"| {fold} | {r['n_sub']} of {r['n_fit']} | {r['n_va']} | {np.sqrt(r['sq1'] / r['n_va']):.1f} | {np.sqrt(r['sq3'] / r['n_va']):.1f} | "
                     f"{r['dt']:.0f} | {r['vram']:.2f} |{' seeded from the printed line of a killed run (RMSE rounded to 0.1 s)' if r.get('seeded') else ''}")
        print(lines[-1], flush=True)
    r1, r3 = np.sqrt(sq1 / n), np.sqrt(sq3 / n)
    gain = r1 - r3
    gates = [("fit plus predict under 30 min (all five folds)", secs / 60, GATES["minutes"], secs / 60 < GATES["minutes"]),
             ("peak VRAM under 16 GB", peak, GATES["vram_gb"], peak < GATES["vram_gb"]),
             ("OOF gain over M1 on this airport of at least 2 s", gain, GATES["gain_s"], gain >= GATES["gain_s"])]
    report = [f"# M3 pilot: TabPFN 6.3.1 on {AIRPORT} lane A (R12)", "",
              f"Train months {TRAIN_MONTHS}, month-pair folds; regenerate with `python -m taxi.m3`. Features {COLS} (M1's set without the constant airport). "
              f"TabPFN sees a seeded subsample of {LIMIT:,} rows per fold (its pretraining limit); M1 is the champion fit on all airports' lane-A rows outside the fold, "
              f"scored on this airport's fold rows. n_estimators 8 (default), device cuda.", "", "## Folds", ""] + lines + ["",
              f"Pooled OOF over {n:,} rows: M1 {r1:.1f} s, M3 {r3:.1f} s, gain {gain:+.1f} s.", "", "## Gates", "",
              "| gate | measured | limit | result |", "|---|---|---|---|"]
    report += [f"| {g} | {v:.1f} | {lim:.0f} | {'PASS' if ok else 'FAIL'} |" for g, v, lim, ok in gates]
    report += ["", f"Pilot verdict: {'all three gates pass; M3 may be declared per airport' if all(ok for *_, ok in gates) else 'a gate fails; M3 stops here unless the declaration is amended'}.", ""]
    text = "\n".join(report) + "\n"
    with open("docs/M3-PILOT.md", "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
