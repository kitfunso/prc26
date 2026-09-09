"""TP2: the TabPFN pilot re-run without M3's two confounds, on one airport.

M3 lost lane A by 32.2 s, but it read M1's 19 columns against a champion that reads 52, and it saw a 50,000-row subsample
against a champion that saw every row. This lane removes both on LIRF before anyone spends the ten-airport budget again.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import torch
from tabpfn import TabPFNRegressor

from taxi.champ import COLS_D2, champion_oof, lane_a
from taxi.data import CATEGORICAL, CV_FOLDS, PLAN_CAT
from taxi.l2 import floored
from taxi.m3 import CHUNK, LIMIT, cached, numeric
from taxi.verdict import rmse

APT = "LIRF"  # the M3-PILOT airport, so the same rows carry the before and after
BAGS = 3  # two disjoint draws already cover ~91% of LIRF's fit pool; the third averages the draw noise
COLS = [c for c in COLS_D2 if c != "apt"]
CATS = tuple(c for c in CATEGORICAL + PLAN_CAT if c in COLS)
CACHE, M3_CACHE = "data/cache/tp2", "data/cache/m3"


def fit_predict(fit: pd.DataFrame, va: pd.DataFrame) -> tuple[np.ndarray, float, float]:
    """BAGS seeded subsamples of the fit rows, each a TabPFN fit on c, predictions averaged and the plan gap added back."""
    torch.cuda.reset_peak_memory_stats()
    t0, preds = time.time(), []
    for seed in range(BAGS):
        sub = fit.sample(n=min(LIMIT, len(fit)), random_state=seed)
        x_fit, x_va, cats = numeric(sub, va, COLS, CATS)
        reg = TabPFNRegressor(device="cuda", categorical_features_indices=cats, random_state=seed)
        reg.fit(x_fit, sub.c.values)
        preds.append(np.concatenate([np.asarray(reg.predict(x_va[i:i + CHUNK])) for i in range(0, len(x_va), CHUNK)]))
        # The allocator's reserve grows across fits until WDDM spills it to system memory (M3: 800 s a fold became 8,000 s).
        del reg
        torch.cuda.empty_cache()
    p = np.mean(preds, axis=0) + va.z_aobt.values
    return floored(p), time.time() - t0, torch.cuda.max_memory_allocated() / 1e9


def oof(tra: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Month-pair OOF over the airport's rows for TP2 and, from M3's own cache, for M3 on those same rows."""
    a = (tra.apt == APT).values
    out, old, lines = np.full(len(tra), np.nan), np.full(len(tra), np.nan), []
    for fold in CV_FOLDS:
        m = tra.month.isin(fold).values
        ix = np.flatnonzero(a & m)
        tag = f"{APT}_{fold[0]:02d}{fold[1]:02d}.parquet"
        c = cached(f"{CACHE}/{tag}", tra.id.values[ix], lambda: fit_predict(tra[a & ~m], tra.iloc[ix]))
        out[ix] = c.pred.values
        prev = f"{M3_CACHE}/{tag}"
        if os.path.exists(prev):
            q = pd.read_parquet(prev)
            if np.array_equal(q.id.values, tra.id.values[ix]):
                old[ix] = q.pred.values
        lines.append(f"| {fold} | {int((a & ~m).sum()):,} | {len(ix):,} | {c.dt.iloc[0] / 60:.0f} | {c.vram.iloc[0]:.2f} |")
        print(lines[-1], flush=True)
    return out, old, lines


def main(out: str = "docs/TP2.md") -> None:
    tra, _ = lane_a()
    p, m3, lines = oof(tra)
    a = np.isfinite(p)
    y, mc2 = tra.y.values, champion_oof("MC2", tra)
    r_tp2, r_mc2 = rmse(p[a] - y[a]), rmse(mc2[a] - y[a])
    ok3 = a & np.isfinite(m3)
    r_m3 = rmse(m3[ok3] - y[ok3]) if ok3.any() else float("nan")

    gate = [f"| beats M3 on the same rows by at least 10 s | {r_m3 - r_tp2:.1f} | 10 | {'PASS' if r_m3 - r_tp2 >= 10 else 'FAIL'} |",
            f"| at or below MC2 on the same rows | {r_mc2 - r_tp2:.1f} | 0 | {'PASS' if r_tp2 <= r_mc2 else 'FAIL'} |"]
    lines_out = [f"# TP2: TabPFN v2.5 on {APT} lane A with D2's columns and no row cap", "",
                 f"Regenerate with `python -m taxi.tp2`. One airport, month-pair OOF only; no holdout fit runs until the gates below pass. "
                 f"TabPFN 6.3.1 at its default model version v2.5, device cuda, n_estimators 8. {len(COLS)} columns, D2's set without the constant "
                 f"airport, against M3's 19. Each fold fits {BAGS} seeded {LIMIT:,}-row subsamples of the airport's fit pool and averages their "
                 f"predictions, so the model class is no longer judged on a quarter of the data. Target is `c` and the plan gap is added back, as in "
                 f"M3; the C1 floor is applied. M3's own cached predictions for these rows are read from `{M3_CACHE}` so the before and after sit on "
                 f"identical rows. This lane cannot promote anything: it decides whether the ten-airport run is worth its budget.", "",
                 "## Folds", "", "| fold | fit pool | predicted rows | min | peak VRAM GB |", "|---|---|---|---|---|", *lines, "",
                 f"Pooled OOF over {int(a.sum()):,} rows: MC2 {r_mc2:.1f} s, M3 {r_m3:.1f} s, TP2 {r_tp2:.1f} s.", "",
                 "## Gates", "", "| gate | measured (s) | limit | result |", "|---|---|---|---|", *gate, "",
                 (f"Both gates pass: the loss recorded against the model class was the inputs, not the class. The ten-airport run is declared next."
                  if r_tp2 <= r_mc2 and r_m3 - r_tp2 >= 10 else
                  f"A gate fails. TabPFN read the champion's own columns on the whole airport and still did not reach it, so the family is closed on "
                  f"lane A and no ten-airport run follows.")]
    text = "\n".join(lines_out) + "\n"
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
