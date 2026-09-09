"""SB3OOF: SB3's own out-of-fold vector, so NN3's blend weight is fitted against the base it is actually applied to.

The shipped weight 0.5189 was fitted out of fold against MC2 and then applied to SB3, which is 2.06 s better on the same holdout
lane. A bag has no out-of-fold vector until one is built: five seeds by five folds, at D2's recorded rounds with no early
stopping, which is exactly how `taxi.sb3.seed_pred` predicts every other row set. Declared as SB3OOF in protocol section 4.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

from taxi.champ import COLS_D2, lane_a
from taxi.data import CV_FOLDS
from taxi.l2 import floored
from taxi.m1 import final_rounds
from taxi.mc1 import predict_cb, stages_cb
from taxi.nn1 import blend_weight
from taxi.sb3 import SEEDS
from taxi.sb3 import predict as sb3_predict
from taxi.verdict import paired_delta, rmse
from taxi.w1 import verdict_line

OOF, MC2_OOF = "data/cache/sb3_oof.parquet", "data/cache/mc2_oof.parquet"
NN3_OOF, NN3_TEA = "data/cache/nn3_oof.parquet", "data/cache/nn3_holdout.parquet"
SHIPPED = 0.5189  # NN3's weight as fitted against MC2 (docs/NN3.md), the number this lane exists to replace
K = 1


def by_id(path: str, ids: np.ndarray) -> np.ndarray:
    return pd.DataFrame({"id": ids}).merge(pd.read_parquet(path), on="id", how="left").pred.to_numpy(float)


def cell(tra: pd.DataFrame, fold: tuple[int, int], seed: int, rounds: tuple[int, int, int]) -> pd.Series:
    """One (fold, seed) fit, cached on disk so a killed run resumes at the next cell rather than at the start."""
    path = f"data/cache/sb3oof_{fold[0]}_{fold[1]}_s{seed}.parquet"
    if not os.path.exists(path):
        m = tra.month.isin(fold).values
        fit, va, t0 = tra[~m], tra[m], time.time()
        pred = predict_cb(stages_cb(fit, iters=rounds, cols=COLS_D2, over={"random_seed": seed}), va, COLS_D2)[0]
        pd.DataFrame({"id": va.id.values, "pred": pred}).to_parquet(path, index=False)
        print(f"fit {fold} seed {seed}: {time.time() - t0:.0f} s", flush=True)
    return pd.read_parquet(path).set_index("id").pred


def build(tra: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """The bag's out-of-fold vector: seeds averaged inside each fold, then C1's floor, matching `taxi.sb3.predict`."""
    rounds, truth = final_rounds("D2", "docs/D2.md"), pd.Series(tra.y.values, index=tra.id.values)
    lines = ["| fold | rows | seed RMSE, best to worst (s) | bagged and floored (s) |", "|---|---|---|---|"]
    out = pd.Series(np.nan, index=tra.id.values)
    for fold in CV_FOLDS:
        seeds = [cell(tra, fold, s, rounds) for s in SEEDS]
        bag = floored(pd.concat(seeds, axis=1).mean(axis=1))
        out.loc[bag.index] = bag.values
        e = sorted(rmse(s.values - truth.loc[s.index].values) for s in seeds)
        lines.append(f"| {fold} | {len(bag):,} | {e[0]:.2f} to {e[-1]:.2f} | "
                     f"{rmse(bag.values - truth.loc[bag.index].values):.2f} |")
        print(lines[-1], flush=True)
    return out.to_numpy(float), lines


def main(out: str = "docs/SB3OOF.md") -> None:
    tra, tea = lane_a()
    oof, lines = build(tra)
    pd.DataFrame({"id": tra.id.values, "pred": oof}).to_parquet(OOF, index=False)

    nn, mc2, y = by_id(NN3_OOF, tra.id.values), by_id(MC2_OOF, tra.id.values), tra.y.values
    ok = np.isfinite(oof) & np.isfinite(nn) & np.isfinite(mc2)
    w = blend_weight(nn[ok], oof[ok], y[ok])

    sb3_te, y_te = sb3_predict(tra, tea), tea.y.values
    nn_te = by_id(NN3_TEA, tea.id.values)
    new, old = floored(w * nn_te + (1 - w) * sb3_te), floored(SHIPPED * nn_te + (1 - SHIPPED) * sb3_te)
    v = paired_delta(sb3_te - y_te, new - y_te, tea.day.values)

    text = "\n".join([
        "# SB3OOF: the blend weight refitted against the base it is applied to", "",
        f"Regenerate with `python -m taxi.sb3oof`. {len(CV_FOLDS)} folds x {len(SEEDS)} seeds = {len(CV_FOLDS) * len(SEEDS)} "
        f"CatBoost fits at D2's recorded rounds {final_rounds('D2', 'docs/D2.md')}, no early stopping, on "
        f"{int(ok.sum()):,} lane-A training rows. Nothing is searched here, so k = {K}.", "",
        "## Every fold", "", *lines, "",
        "## What the right base changes", "",
        "| quantity | base MC2 (shipped) | base SB3 (here) |", "|---|---|---|",
        f"| weight on NN3, fitted out of fold | {SHIPPED:.4f} | {w:.4f} |",
        f"| out-of-fold RMSE of the base | {rmse(mc2[ok] - y[ok]):.2f} | {rmse(oof[ok] - y[ok]):.2f} |",
        f"| holdout RMSE of the blend | {rmse(old - y_te):.2f} | {rmse(new - y_te):.2f} |", "",
        f"SB3 alone reads {rmse(sb3_te - y_te):.2f} s on the same holdout rows.", "",
        verdict_line("SB3+NN3 at the SB3-fitted weight", "SB3", v, k=K)]) + "\n"
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
