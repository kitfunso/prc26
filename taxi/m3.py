"""M3 (R12, amended 2026-09-05): TabPFN regressor per airport on lane A, all ten airports; month-pair OOF and one holdout fit per airport, cached
fold by fold; compared with MC2 on the same rows and reported, never a verdict on its own (E1 blends it)."""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import torch
from tabpfn import TabPFNRegressor

from taxi.champ import champion_holdout, champion_oof, lane_a
from taxi.data import CATEGORICAL, CV_FOLDS, TRAIN_MONTHS
from taxi.m1 import MODEL_FEATURES
from taxi.verdict import rmse

LIMIT = 50_000  # TabPFN 6.3.1 pretraining limit on rows; a seeded subsample of the airport's fit rows
CHUNK = 4_096  # predict in slices: one 23k-row predict held 11 GB of RAM and the harness killed the run
COLS = [c for c in MODEL_FEATURES if c != "apt"]
CACHE = "data/cache/m3"
HOLDOUT = "data/cache/m3_holdout.parquet"


def numeric(fit: pd.DataFrame, va: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Categoricals as integer codes shared by both frames (missing stays NaN); TabPFN takes those columns by index."""
    both = pd.concat([fit[COLS], va[COLS]]).reset_index(drop=True)
    for c in CATEGORICAL:
        codes = pd.Categorical(both[c]).codes.astype(float)
        both[c] = np.where(codes < 0, np.nan, codes)
    x = both.values.astype(float)
    return x[: len(fit)], x[len(fit):], [COLS.index(c) for c in CATEGORICAL]


def fit_predict(fit: pd.DataFrame, va: pd.DataFrame) -> tuple[np.ndarray, float, float]:
    """One airport: TabPFN on a seeded subsample of the fit rows, predicted on va in chunks; returns predictions of y, seconds and peak VRAM GB."""
    sub = fit.sample(n=min(LIMIT, len(fit)), random_state=0)
    x_fit, x_va, cats = numeric(sub, va)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    reg = TabPFNRegressor(device="cuda", categorical_features_indices=cats, random_state=0)
    reg.fit(x_fit, sub.c.values)
    p = np.concatenate([np.asarray(reg.predict(x_va[i:i + CHUNK])) for i in range(0, len(x_va), CHUNK)]) + va.z_aobt.values
    dt, vram = time.time() - t0, torch.cuda.max_memory_allocated() / 1e9
    # Held across fits, the allocator's reserve grows until WDDM spills it to system memory: 800 s a fold became 8,000 s at fit 16.
    del reg
    torch.cuda.empty_cache()
    return p, dt, vram


def cached(path: str, ids: np.ndarray, make) -> pd.DataFrame:
    if os.path.exists(path):
        c = pd.read_parquet(path)
        if np.array_equal(c.id.values, ids):
            return c
    p, dt, vram = make()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    c = pd.DataFrame({"id": ids, "pred": p, "dt": dt, "vram": vram})
    c.to_parquet(path, index=False)
    return c


def oof(tra: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Month-pair OOF predictions for every lane-A training row, one TabPFN fit per (airport, fold), each cached under data/cache/m3/."""
    out, lines = np.full(len(tra), np.nan), []
    for apt in sorted(tra.apt.unique()):
        a = (tra.apt == apt).values
        for fold in CV_FOLDS:
            m = tra.month.isin(fold).values
            ix = np.flatnonzero(a & m)
            c = cached(f"{CACHE}/{apt}_{fold[0]:02d}{fold[1]:02d}.parquet", tra.id.values[ix], lambda: fit_predict(tra[a & ~m], tra.iloc[ix]))
            out[ix] = c.pred.values
            lines.append(f"| {apt} | {fold} | {int((a & ~m).sum()):,} | {len(ix):,} | {c.dt.iloc[0]:.0f} | {c.vram.iloc[0]:.2f} |")
            print(lines[-1], flush=True)
    return out, lines


def holdout(tra: pd.DataFrame, tea: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """One TabPFN fit per airport on the ten training months' subsample, predicted on the airport's Jan+Jul rows; cached under data/cache/m3/."""
    out, lines = np.full(len(tea), np.nan), []
    for apt in sorted(tea.apt.unique()):
        a, ix = (tra.apt == apt).values, np.flatnonzero((tea.apt == apt).values)
        c = cached(f"{CACHE}/{apt}_holdout.parquet", tea.id.values[ix], lambda: fit_predict(tra[a], tea.iloc[ix]))
        out[ix] = c.pred.values
        lines.append(f"| {apt} | holdout | {int(a.sum()):,} | {len(ix):,} | {c.dt.iloc[0]:.0f} | {c.vram.iloc[0]:.2f} |")
        print(lines[-1], flush=True)
    pd.DataFrame({"id": tea.id.values, "pred": out}).to_parquet(HOLDOUT, index=False)
    return out, lines


def main() -> None:
    tra, tea = lane_a()
    o3, lines_o = oof(tra)
    h3, lines_h = holdout(tra, tea)
    o2, h2 = champion_oof("MC2", tra), champion_holdout("MC2", tra, tea)
    by_apt = ["| airport | OOF rows | MC2 OOF (s) | M3 OOF (s) | gain | holdout rows | MC2 holdout (s) | M3 holdout (s) | gain |", "|---|---|---|---|---|---|---|---|---|"]
    for apt in sorted(tra.apt.unique()):
        a, b = (tra.apt == apt).values, (tea.apt == apt).values
        r2, r3 = rmse(o2[a] - tra.y.values[a]), rmse(o3[a] - tra.y.values[a])
        s2, s3 = rmse(h2[b] - tea.y.values[b]), rmse(h3[b] - tea.y.values[b])
        by_apt.append(f"| {apt} | {int(a.sum()):,} | {r2:.1f} | {r3:.1f} | {r2 - r3:+.1f} | {int(b.sum()):,} | {s2:.1f} | {s3:.1f} | {s2 - s3:+.1f} |")
    secs = sum(float(l.split("|")[5]) for l in lines_o + lines_h)
    vram = max(float(l.split("|")[6]) for l in lines_o + lines_h)
    report = ["# M3: TabPFN 6.3.1 per airport on lane A, all ten airports (R12, amended 2026-09-05)", "",
              f"Train months {TRAIN_MONTHS}, month-pair folds; regenerate with `python -m taxi.m3`. Features {COLS} (M1's set without the constant airport). "
              f"Each fit sees a seeded subsample of {LIMIT:,} of the airport's fit rows (the model's pretraining limit), n_estimators 8 (default), device cuda. "
              f"MC2 is the champion's cached month-pair OOF and holdout refit on the same rows (`taxi.champ`). Reported, never a verdict: E1 blends M3 with MC2. "
              f"The pilot's 30 min time gate was lifted by Keith's instruction of 2026-09-05 (amendment log); the VRAM gate (16 GB) is still reported.", "",
              "## By airport", "", *by_apt, "",
              f"Pooled: OOF MC2 {rmse(o2 - tra.y.values):.1f} s, M3 {rmse(o3 - tra.y.values):.1f} s; holdout MC2 {rmse(h2 - tea.y.values):.1f} s, "
              f"M3 {rmse(h3 - tea.y.values):.1f} s. Wall clock {secs / 3600:.1f} h over {len(lines_o) + len(lines_h)} fits; peak VRAM {vram:.2f} GB "
              f"({'under' if vram < 16 else 'OVER'} the 16 GB gate).", "",
              "## Fits", "", "| airport | fold | fit rows | predicted rows | fit + predict s | peak VRAM GB |", "|---|---|---|---|---|---|", *lines_o, *lines_h]
    text = "\n".join(report) + "\n"
    with open("docs/M3.md", "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
