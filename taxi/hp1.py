"""HP1: the champion's own cell, the one axis never tuned. Eleven one-knob arms off D2 plus a composed twelfth, k = 12.

COVERAGE section 6 skipped the optimiser "while model classes and inputs were untried"; they are tried now, so the cell is what is left.
"""
from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np
import pandas as pd

from taxi.champ import COLS_D2, lane_a
from taxi.l2 import floored
from taxi.m2 import is_tail
from taxi.mc1 import CELL, cv_a, predict_cb, stages_cb
from taxi.sb3 import SEEDS
from taxi.sb3 import predict as sb3_predict
from taxi.verdict import paired_delta, rmse, score
from taxi.w1 import verdict_line

BASE_CV, BASE = 211.8, "D2"  # docs/D2.md, already measured and therefore the reference rather than a trial
ARMS: dict[str, dict] = {
    "depth6": {"depth": 6},
    "depth10": {"depth": 10},
    "lr05": {"learning_rate": 0.05},
    "l2reg1": {"l2_leaf_reg": 1.0},
    "l2reg10": {"l2_leaf_reg": 10.0},
    "l2reg30": {"l2_leaf_reg": 30.0},
    "rstr4": {"random_strength": 4.0},
    "bagt0": {"bagging_temperature": 0.0},
    "bagt3": {"bagging_temperature": 3.0},
    "onehot64": {"one_hot_max_size": 64},
    "border128": {"border_count": 128},
}
K = len(ARMS) + 1
CACHE = "data/cache/hp1_arms.json"


def arms_cv(tra: pd.DataFrame) -> dict[str, dict]:
    """Each arm's five-fold CV, cached to json so a killed run resumes at the next arm; a raising arm is recorded, never dropped."""
    got = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    for name, over in ARMS.items():
        if name in got:
            continue
        t0 = time.time()
        try:
            c = cv_a(tra, COLS_D2, f"HP1-{name}", ref=BASE, over=over)
            got[name] = {"over": over, "cv": c.rmse, "rounds": list(c.rounds), "min": (time.time() - t0) / 60}
        except Exception as exc:
            got[name] = {"over": over, "cv": None, "error": f"{type(exc).__name__}: {exc}"[:200], "min": (time.time() - t0) / 60}
            print(f"HP1-{name} raised: {got[name]['error']}", flush=True)
        json.dump(got, open(CACHE, "w"), indent=1)
    return got


def composed(got: dict[str, dict]) -> dict:
    """Every knob that beat the base, best arm per knob; two arms of one knob never both enter."""
    best: dict[str, tuple[float, object]] = {}
    for name, r in got.items():
        if name == "composed" or r.get("cv") is None or r["cv"] >= BASE_CV:
            continue
        (knob, val), = r["over"].items()
        if knob not in best or r["cv"] < best[knob][0]:
            best[knob] = (r["cv"], val)
    return {k: v for k, (_, v) in best.items()}


def seed_pred(seed: int, tra: pd.DataFrame, tea: pd.DataFrame, rounds: tuple[int, int, int], over: dict, key: str) -> np.ndarray:
    path = f"data/cache/hp1_{key}_seed{seed}.parquet"
    if os.path.exists(path):
        c = pd.read_parquet(path)
        if len(c) == len(tea) and np.array_equal(c.id.values, tea.id.values):
            return c.pred.values
    t0 = time.time()
    pred = predict_cb(stages_cb(tra, iters=rounds, cols=COLS_D2, over={**over, "random_seed": seed}), tea, COLS_D2)[0]
    pd.DataFrame({"id": tea.id.values, "pred": pred}).to_parquet(path, index=False)
    print(f"seed {seed}: {time.time() - t0:.0f} s{score(pred, tea)}", flush=True)
    return pred


def main(dst: str = "docs/HP1.md") -> None:
    tra, tea = lane_a()
    got = arms_cv(tra)
    comp = composed(got)
    if comp and comp not in [r["over"] for r in got.values()] and "composed" not in got:
        t0 = time.time()
        c = cv_a(tra, COLS_D2, "HP1-composed", ref=BASE, over=comp)
        got["composed"] = {"over": comp, "cv": c.rmse, "rounds": list(c.rounds), "min": (time.time() - t0) / 60}
        json.dump(got, open(CACHE, "w"), indent=1)

    ran = {n: r for n, r in got.items() if r.get("cv") is not None}
    win = min(ran, key=lambda n: ran[n]["cv"]) if ran else None
    beats = win is not None and ran[win]["cv"] < BASE_CV
    rows = [f"| {n} | `{json.dumps(r['over'])}` | {r['cv']:.1f} | {r['cv'] - BASE_CV:+.1f} | {r.get('rounds')} | {r['min']:.0f} |"
            if r.get("cv") is not None else f"| {n} | `{json.dumps(r['over'])}` | raised | | {r.get('error', '')} | {r['min']:.0f} |"
            for n, r in got.items()]

    out = [f"# HP1: tuning the champion's cell, Jan + Jul 2025 development holdout", "",
           f"Regenerate with `python -m taxi.hp1`. Eleven cells, each ONE knob off D2's cell on its {len(COLS_D2)} columns, plus a twelfth "
           f"composing whatever beat the base; k = {K}. The base is D2's own CV of {BASE_CV} s (`docs/D2.md`), already measured and so the "
           f"reference rather than a trial. Every CatBoost model in this campaign shares `taxi.mc1.CELL` and no knob in it had been moved.", "",
           "## Month-pair CV by arm", "",
           "| arm | knob | CV RMSE (s) | vs D2 | trees (clf, tail, body) | min |", "|---|---|---|---|---|---|", *rows, "",
           (f"CV winner: {win} at {ran[win]['cv']:.1f} s." if beats else
            f"Best arm: {win} at {ran[win]['cv']:.1f} s, which does not beat the base.") if win else "Every arm raised; no winner."]

    if not beats:
        out += ["", f"No arm beats D2's {BASE_CV} s on CV, so nothing meets the holdout and SB3 stands. The cell is measured and closed: "
                    f"the hand-set values were not leaving anything on the table that one knob at a time can reach."]
    else:
        over, rounds, y = ran[win]["over"], tuple(ran[win]["rounds"]), tea.y.values
        key = hashlib.sha256(json.dumps(over, sort_keys=True).encode()).hexdigest()[:10]
        bag = floored(np.mean([seed_pred(s, tra, tea, rounds, over, key) for s in SEEDS], axis=0))
        e0, e1 = sb3_predict(tra, tea) - y, bag - y
        v = paired_delta(e0, e1, tea.day.values)
        vt, jan, jul = is_tail(tea), tea.month.values == 1, tea.month.values == 7
        row = lambda n, e: f"| {n} | {rmse(e):.2f} | {rmse(e[jan]):.1f} | {rmse(e[jul]):.1f} | {rmse(e[vt]):.1f} | {rmse(e[~vt]):.1f} |"  # noqa: E731
        out += ["", f"## Holdout verdict vs SB3 ({len(tea):,} rows), the winner as its own five-seed bag", "",
                f"SB3 is a five-seed bag and `docs/SEEDS.md` prices the bag at 1.07 s over one seed, so the challenger is bagged the same way "
                f"at cell `{json.dumps(over)}` and trees {rounds}.", "",
                "| model | RMSE | Jan | Jul | tail rows | body rows |", "|---|---|---|---|---|---|",
                row("SB3", e0), row(f"HP1-{win} bag", e1), "",
                verdict_line(f"HP1-{win}", "SB3", v, k=K)]

    text = "\n".join(out) + "\n"
    with open(dst, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
