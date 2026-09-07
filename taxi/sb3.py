"""SB3: the five-seed D2 bag, floored.

As promoted at H52 this also carried ST1a's gate, which set each row's blend weight from `is_tail(te)`. That reads `c = y - z_aobt`, the
predicted row's own label, so the 229.7752 s verdict was scored with the answer in hand and the model could not run on the ranking file at
all. The gate is removed here (H53) and what is left is the bag, which is honest and predicts anywhere. D2 is deliberately NOT registered
in CHAMPIONS_A: that registry feeds `_fingerprint`, and adding its columns would invalidate every cached champion prediction.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

from taxi.champ import COLS_D2, _fingerprint, champion_holdout, lane_a, refit_predict
from taxi.l2 import floored
from taxi.m1 import final_rounds
from taxi.m2 import is_tail
from taxi.mc1 import predict_cb, stages_cb
from taxi.verdict import paired_delta, rmse, score
from taxi.w1 import verdict_line

SEEDS = (0, 1, 2, 3, 4)
EXTRA = ("d_as", "occ", "t_occ")
W_TAIL, W_BODY = 1.0, 0.7
K = 1


def _fp(tra: pd.DataFrame, tea: pd.DataFrame) -> int:
    """The shared fingerprint widened by D2's three extra columns, which no registered champion reads."""
    h = sum(int(pd.util.hash_pandas_object(d[list(EXTRA)], index=False).sum()) for d in (tra, tea))
    return int((_fingerprint(tra, tea) + h) % (1 << 62))


def seed_pred(seed: int, tra: pd.DataFrame, tea: pd.DataFrame, rounds: tuple[int, int, int], fp: int,
              tag: str = "holdout") -> np.ndarray:
    """One D2 seed's predictions on tea, cached so a killed run resumes at the next seed; tag keeps each row set's caches apart."""
    path = f"data/cache/sb3_seed{seed}_{tag}.parquet"
    if os.path.exists(path):
        c = pd.read_parquet(path)
        if len(c) == len(tea) and c.fp.iloc[0] == fp and np.array_equal(c.id.values, tea.id.values):
            return c.pred.values
    t0 = time.time()
    pred = predict_cb(stages_cb(tra, iters=rounds, cols=COLS_D2, over={"random_seed": seed}), tea, COLS_D2)[0]
    pd.DataFrame({"id": tea.id.values, "pred": pred, "fp": fp}).to_parquet(path, index=False)
    print(f"seed {seed}: {time.time() - t0:.0f} s{score(pred, tea)}", flush=True)
    return pred


def predict(tr: pd.DataFrame, te: pd.DataFrame, tag: str = "holdout") -> np.ndarray:
    """SB3 on any row set: the five-seed D2 bag under C1's floor, reading nothing but features."""
    rounds, fp = final_rounds("D2", "docs/D2.md"), _fp(tr, te)
    return floored(np.mean([seed_pred(s, tr, te, rounds, fp, tag) for s in SEEDS], axis=0))


def main() -> None:
    tra, tea = lane_a()
    rounds, fp, y = final_rounds("D2", "docs/D2.md"), _fp(tra, tea), tea.y.values
    preds = [seed_pred(s, tra, tea, rounds, fp) for s in SEEDS]
    b = np.mean(preds, axis=0)
    base = floored(champion_holdout("MC2", tra, tea))
    gated = floored(np.where(is_tail(tea), W_TAIL, W_BODY) * b
                    + (1 - np.where(is_tail(tea), W_TAIL, W_BODY)) * champion_holdout("D1", tra, tea))
    steps = {"MC2 floored (predecessor)": base, "D2 seed 0 floored": floored(preds[0]), "D2 seed bag floored (SB3)": predict(tra, tea),
             "the same bag under ST1a's withdrawn label gate": gated}
    sb3 = steps["D2 seed bag floored (SB3)"]
    v = paired_delta(base - y, sb3 - y, tea.day.values)
    vt, jan, jul = is_tail(tea), tea.month.values == 1, tea.month.values == 7
    row = lambda n, p: f"| {n} | {rmse(p - y):.2f} | {rmse((p - y)[jan]):.1f} | {rmse((p - y)[jul]):.1f} | {rmse((p - y)[vt]):.1f} | {rmse((p - y)[~vt]):.1f} |"  # noqa: E731
    spread = max(rmse(p - y) for p in preds) - min(rmse(p - y) for p in preds)
    seeds = [f"| {s} | {rmse(p - y):.2f} | {paired_delta(base - y, floored(p) - y, tea.day.values).delta:+.2f} |" for s, p in zip(SEEDS, preds)]

    out = [f"# SB3: the five-seed D2 bag, Jan + Jul 2025 development holdout", "",
           f"Regenerate with `python -m taxi.sb3`. D2's cell and recorded trees {rounds} on its {len(COLS_D2)} columns, fit once per seed on the ten "
           f"training months with no early stopping, then C1's floor at 0 s. Every constant was fixed by an earlier declared run, so k = {K}. "
           f"ST1a's gate was part of SB3 as promoted at H52 and is withdrawn: it set each row's weight from `is_tail(te)`, which reads "
           f"`c = y - z_aobt`, that row's own label. Its line is kept below to price what the withdrawal costs. The verdict line reads NOT "
           f"promoted because SB3 already holds lane A: this is H52's promotion re-scored without the gate, not a new one.", "",
           "## Seeds (unfloored RMSE; the delta is each seed floored, against the champion)", "",
           "| seed | holdout RMSE (s) | paired delta vs MC2 floored (s) |", "|---|---|---|", *seeds, "",
           f"Seed-to-seed RMSE spread: {spread:.2f} s.", "",
           f"## Each step added, on the same {len(tea):,} rows", "",
           "| model | RMSE | Jan | Jul | tail rows | body rows |", "|---|---|---|---|---|---|",
           *[row(n, p) for n, p in steps.items()], "",
           verdict_line("SB3", "MC2", v, k=K)]
    text = "\n".join(out) + "\n"
    with open("docs/SB3.md", "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
