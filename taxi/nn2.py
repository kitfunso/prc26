"""NN2: NN1's cell, bagged over seeds and trained to the end of its schedule. Protocol section 4, k = 1, no search.

NN1 reached 239.38 s at 0.9169 error agreement with SB3 and `docs/BF.md`'s frontier asks for under 0.905 at that level, so the
blend banked +1.40 s against a 2 s floor. Two facts say the net, not the family, is what fell short: every fold stopped at epoch
17 to 21 of a 60-epoch cosine schedule, so the learning rate never annealed, and SB3 is a five-seed bag while NN1 was one seed.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

from taxi.bf import agreement, best_blend
from taxi.champ import lane_a
from taxi.data import CV_FOLDS
from taxi.l2 import floored
from taxi.nn1 import CELL, base_oof, blend_weight, predict, train
from taxi.sb3 import SEEDS
from taxi.sb3 import predict as sb3_predict
from taxi.verdict import paired_delta, rmse
from taxi.w1 import verdict_line

PATIENCE = 15
OOF_CACHE, TEA_CACHE = "data/cache/nn2_oof.parquet", "data/cache/nn2_holdout.parquet"


def _cached(path: str, ids: np.ndarray) -> np.ndarray | None:
    if not os.path.exists(path):
        return None
    d = pd.read_parquet(path)
    return d.set_index("id").pred.reindex(ids).to_numpy(float) if len(d) == len(ids) else None


def _save(path: str, ids: np.ndarray, pred: np.ndarray) -> None:
    pd.DataFrame({"id": ids, "pred": pred}).to_parquet(path, index=False)


def bagged_oof(tra: pd.DataFrame, cell: dict | None = None, patience: int | None = PATIENCE,
               cache: str = OOF_CACHE) -> tuple[np.ndarray, list[str]]:
    """One bagged out-of-fold vector: every fold predicted by the mean of len(SEEDS) nets, each held out of that fold."""
    got = _cached(cache, tra.id.values)
    if got is not None:
        return got, ["| cached | | | |"]
    oof, lines = np.full(len(tra), np.nan), ["| fold | bagged RMSE (s) | epochs per seed | fit s |", "|---|---|---|---|"]
    for fold in CV_FOLDS:
        m, t0 = tra.month.isin(fold).values, time.time()
        va, eps = tra[m].reset_index(drop=True), []
        preds = []
        for s in SEEDS:
            model, enc, mu, sd, ep = train(tra[~m].reset_index(drop=True), seed=s, patience=patience, cell=cell)
            preds.append(predict(model, enc, va, mu, sd))
            eps.append(ep)
        oof[m] = np.mean(preds, axis=0)
        lines.append(f"| {fold} | {rmse(oof[m] - va.y.values):.1f} | {eps} | {time.time() - t0:.0f} |")
        print(lines[-1], flush=True)
    _save(cache, tra.id.values, oof)
    return oof, lines


def bagged_holdout(tra: pd.DataFrame, tea: pd.DataFrame, cell: dict | None = None,
                   patience: int | None = PATIENCE, cache: str = TEA_CACHE) -> np.ndarray:
    got = _cached(cache, tea.id.values)
    if got is not None:
        return got
    preds = []
    for s in SEEDS:
        t0 = time.time()
        model, enc, mu, sd, ep = train(tra, seed=s, patience=patience, cell=cell)
        preds.append(predict(model, enc, tea, mu, sd))
        print(f"holdout seed {s}: {rmse(preds[-1] - tea.y.values):.2f} s, {ep} epochs, {time.time() - t0:.0f} s", flush=True)
    p = np.mean(preds, axis=0)
    _save(cache, tea.id.values, p)
    return p


def main(out: str = "docs/NN2.md") -> None:
    tra, tea = lane_a()
    t0 = time.time()
    oof, fold_lines = bagged_oof(tra)
    base = base_oof(tra)
    ok = np.isfinite(oof) & np.isfinite(base)
    w = blend_weight(oof[ok], base[ok], tra.y.values[ok])
    print(f"OOF rows {int(ok.sum()):,}; weight on NN2 fitted out of fold: {w:.4f}", flush=True)

    nn_te = floored(bagged_holdout(tra, tea))
    sb3_te, y = sb3_predict(tra, tea), tea.y.values
    e0, ex = sb3_te - y, nn_te - y
    blend = floored(w * nn_te + (1 - w) * sb3_te)
    v = paired_delta(e0, blend - y, tea.day.values)
    ow, orm, rho = best_blend(e0, ex)
    jan, jul = tea.month.values == 1, tea.month.values == 7
    row = lambda n, e: f"| {n} | {rmse(e):.2f} | {rmse(e[jan]):.2f} | {rmse(e[jul]):.2f} |"  # noqa: E731

    report = [
        "# NN2: NN1 bagged and trained out, Jan + Jul 2025 development holdout", "",
        f"Regenerate with `python -m taxi.nn2`. Declared in protocol section 4 before the code was written; k = 1, no search. "
        f"NN1's cell unchanged ({CELL}) except patience {CELL['patience']} to {PATIENCE}, bagged over seeds {list(SEEDS)} at both "
        f"stages. Wall {(time.time() - t0) / 60:.1f} min.", "",
        "## Out-of-fold folds", "", *fold_lines, "",
        f"Weight on NN2, fitted out of fold against MC2's cached out-of-fold predictions on {int(ok.sum()):,} rows: **{w:.4f}**. "
        f"The holdout is never read to choose it. The base is MC2 and the challenger blends against SB3, an inherited "
        f"mis-specification from NN1 priced there at 0.01 s (fitted 0.2862 against an oracle 0.264).", "",
        f"## Holdout ({len(tea):,} rows)", "",
        "| model | RMSE | Jan | Jul |", "|---|---|---|---|",
        row("SB3 (champion)", e0), row("NN2 alone (diagnostic)", ex), row(f"SB3 + NN2 at w={w:.4f} (challenger)", blend - y), "",
        f"NN2 reaches {rmse(ex):.2f} s at error agreement **{rho:.4f}** with SB3, against NN1's 239.38 s at 0.9169. The frontier "
        f"in `docs/BF.md` asks for agreement under {0.905 if rmse(ex) > 237 else 0.931} near this level. Oracle weight "
        f"{ow:.3f} on SB3 reaching {orm:.2f} s, a gain of {rmse(e0) - orm:+.2f} s.", "",
        verdict_line("SB3 + NN2", "SB3", v),
    ]
    text = "\n".join(report) + "\n"
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
