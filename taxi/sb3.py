"""SB3: SB2's composition on D2's base instead of MC2's.

SB2 composed three parked ties on MC2 and landed +1.3 s, still under the 2 s floor. D2 is a fourth parked tie, +1.0 s over MC2 on three extra
columns, and it is the base the other three were never tried on. Every constant here was fixed by an earlier declared run, so k = 1 again.
D2 is deliberately NOT registered in CHAMPIONS_A: that registry feeds `_fingerprint`, and adding its columns would invalidate every cached
champion prediction this campaign has paid for.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

from taxi.champ import COLS_D2, _fingerprint, champion_holdout, lane_a
from taxi.l2 import floored
from taxi.m1 import final_rounds
from taxi.m2 import is_tail
from taxi.mc1 import predict_cb, stages_cb
from taxi.verdict import paired_delta, rmse
from taxi.w1 import verdict_line

SEEDS = (0, 1, 2, 3, 4)
EXTRA = ("d_as", "occ", "t_occ")
W_TAIL, W_BODY = 1.0, 0.7
K = 1


def _fp(tra: pd.DataFrame, tea: pd.DataFrame) -> int:
    """The shared fingerprint widened by D2's three extra columns, which no registered champion reads."""
    h = sum(int(pd.util.hash_pandas_object(d[list(EXTRA)], index=False).sum()) for d in (tra, tea))
    return int((_fingerprint(tra, tea) + h) % (1 << 62))


def seed_pred(seed: int, tra: pd.DataFrame, tea: pd.DataFrame, rounds: tuple[int, int, int], fp: int) -> np.ndarray:
    """One D2 seed's holdout predictions, cached so a killed run resumes at the next seed."""
    path = f"data/cache/sb3_seed{seed}_holdout.parquet"
    if os.path.exists(path):
        c = pd.read_parquet(path)
        if len(c) == len(tea) and c.fp.iloc[0] == fp and np.array_equal(c.id.values, tea.id.values):
            return c.pred.values
    t0 = time.time()
    pred = predict_cb(stages_cb(tra, iters=rounds, cols=COLS_D2, over={"random_seed": seed}), tea, COLS_D2)[0]
    pd.DataFrame({"id": tea.id.values, "pred": pred, "fp": fp}).to_parquet(path, index=False)
    print(f"seed {seed}: {time.time() - t0:.0f} s, RMSE {rmse(pred - tea.y.values):.2f}", flush=True)
    return pred


def main() -> None:
    tra, tea = lane_a()
    rounds, fp, y = final_rounds("D2", "docs/D2.md"), _fp(tra, tea), tea.y.values
    preds = [seed_pred(s, tra, tea, rounds, fp) for s in SEEDS]
    b, hd = np.mean(preds, axis=0), champion_holdout("D1", tra, tea)
    base = floored(champion_holdout("MC2", tra, tea))
    w = np.where(is_tail(tea), W_TAIL, W_BODY)
    steps = {"MC2 floored (champion)": base, "D2 seed 0 floored": floored(preds[0]), "D2 seed bag": floored(b),
             "D2 seed bag + ST1a gate (SB3)": floored(w * b + (1 - w) * hd)}
    sb3 = steps["D2 seed bag + ST1a gate (SB3)"]
    v = paired_delta(base - y, sb3 - y, tea.day.values)
    vt, jan, jul = is_tail(tea), tea.month.values == 1, tea.month.values == 7
    row = lambda n, p: f"| {n} | {rmse(p - y):.2f} | {rmse((p - y)[jan]):.1f} | {rmse((p - y)[jul]):.1f} | {rmse((p - y)[vt]):.1f} | {rmse((p - y)[~vt]):.1f} |"  # noqa: E731
    spread = max(rmse(p - y) for p in preds) - min(rmse(p - y) for p in preds)
    seeds = [f"| {s} | {rmse(p - y):.2f} | {paired_delta(base - y, floored(p) - y, tea.day.values).delta:+.2f} |" for s, p in zip(SEEDS, preds)]

    out = [f"# SB3: SB2's composition on D2's base, Jan + Jul 2025 development holdout", "",
           f"Regenerate with `python -m taxi.sb3`. D2's cell and recorded trees {rounds} on its {len(COLS_D2)} columns, fit once per seed on the ten "
           f"training months with no early stopping, then ST1a's out-of-fold weights ({W_TAIL} on tail rows, {W_BODY} on body rows, blending toward the "
           f"cached D1 holdout prediction) and C1's floor at 0 s. Every constant was fixed by an earlier declared run, so k = {K}.", "",
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
