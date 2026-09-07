"""SB1: five-seed bag of MC2's cell at its recorded rounds; seed 0 is the cached MC2 refit, seeds 1 to 4 are fit here; k = 1 against MC2."""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

from taxi.champ import CHAMPIONS_A, _fingerprint, champion_holdout, lane_a
from taxi.m1 import final_rounds
from taxi.m2 import is_tail
from taxi.mc1 import predict_cb, stages_cb
from taxi.verdict import paired_delta, rmse, score
from taxi.w1 import verdict_line

NAME, SEEDS, K = "MC2", (1, 2, 3, 4), 1


def seed_path(seed: int) -> str:
    return f"data/cache/sb1_seed{seed}_holdout.parquet"


def seed_pred(seed: int, tra: pd.DataFrame, tea: pd.DataFrame, rounds: tuple[int, int, int], cols: list[str], fp: int) -> np.ndarray:
    """One seed's holdout predictions, cached so a killed run resumes at the next seed."""
    path = seed_path(seed)
    if os.path.exists(path):
        c = pd.read_parquet(path)
        if c.fp.iloc[0] == fp and np.array_equal(c.id.values, tea.id.values):
            return c.pred.values
    t0 = time.time()
    pred = predict_cb(stages_cb(tra, iters=rounds, cols=cols, over={"random_seed": seed}), tea, cols)[0]
    pd.DataFrame({"id": tea.id.values, "pred": pred, "fp": fp}).to_parquet(path, index=False)
    print(f"seed {seed}: {time.time() - t0:.0f} s{score(pred, tea)}", flush=True)
    return pred


def main() -> None:
    tra, tea = lane_a()
    cols, tag, doc, _ = CHAMPIONS_A[NAME]
    rounds, fp, y = final_rounds(tag, doc), _fingerprint(tra, tea), tea.y.values
    preds = [champion_holdout(NAME, tra, tea)] + [seed_pred(s, tra, tea, rounds, cols, fp) for s in SEEDS]
    bag = np.mean(preds, axis=0)
    e0, e1 = preds[0] - y, bag - y
    v = paired_delta(e0, e1, tea.day.values)
    vt, jan, jul = is_tail(tea), tea.month.values == 1, tea.month.values == 7
    row = lambda name, e: f"| {name} | {rmse(e):.1f} | {rmse(e[jan]):.1f} | {rmse(e[jul]):.1f} | {rmse(e[vt]):.1f} | {rmse(e[~vt]):.1f} |"
    seeds = [f"| {s} | {rmse(p - y):.2f} | {paired_delta(e0, p - y, tea.day.values).delta:+.2f} |" for s, p in zip((0, *SEEDS), preds)]
    spread = max(rmse(p - y) for p in preds) - min(rmse(p - y) for p in preds)
    report = [f"# SB1: five-seed bag of {NAME}'s cell, Jan+Jul 2025 development holdout", "",
              f"Regenerate with `python -m taxi.sb1`. {NAME}'s cell and recorded rounds {rounds} on its {len(cols)} columns, fit once per seed on the ten "
              f"training months with no early stopping; seed 0 is the cached {NAME} refit. The challenger is the mean of the five predictions; k = {K}.", "",
              "## Seeds (the spread is the GPU reproducibility yardstick)", "",
              "| seed | holdout RMSE (s) | paired delta vs seed 0 (s) |", "|---|---|---|"] + seeds + ["", f"Seed-to-seed RMSE spread: {spread:.2f} s.", "",
              f"## Holdout verdict vs champion {NAME} ({len(tea)} rows; tail rows {int(vt.sum())})", "",
              "| model | RMSE | Jan | Jul | tail rows | body rows |", "|---|---|---|---|---|---|", row(NAME, e0), row("SB1", e1), "",
              verdict_line("SB1", NAME, v, k=K)]
    text = "\n".join(report) + "\n"
    with open("docs/SB1.md", "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
