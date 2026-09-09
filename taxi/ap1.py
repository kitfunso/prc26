"""AP1: ten CatBoost models, one per airport, against the pooled model that reads the airport as a categorical.

Every lane-A GBM in this campaign pools the ten airports and hands `apt` over as a categorical, so any split, interaction or
tail threshold that differs by airport is paid for out of tree depth. B1, B4 and M3 all fit per airport; no GBM ever has.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

from taxi.champ import COLS_D2, lane_a
from taxi.data import CV_FOLDS
from taxi.l2 import floored
from taxi.m2 import is_tail
from taxi.mc1 import predict_cb, stages_cb
from taxi.sb3 import SEEDS
from taxi.sb3 import predict as sb3_predict
from taxi.verdict import paired_delta, rmse, score
from taxi.w1 import verdict_line

BASE_CV, BASE = 211.8, "D2"  # docs/D2.md, the pooled cell this lane changes one thing about
COLS = [c for c in COLS_D2 if c != "apt"]  # constant inside every fit
CACHE, OOF_CACHE, TEA_CACHE = "data/cache/ap1", "data/cache/ap1_oof.parquet", "data/cache/ap1_holdout.parquet"


def oof(tra: pd.DataFrame) -> tuple[np.ndarray, dict[str, list[int]], list[str]]:
    """One fit per airport per fold, early-stopped on the fold exactly as `taxi.mc1.cv_a` does for the pooled model.
    A fold's fifty predictions and its fifty tree counts are cached together, so a killed run resumes at the fold boundary."""
    apts = sorted(tra.apt.unique())
    os.makedirs(CACHE, exist_ok=True)
    out, trees = np.full(len(tra), np.nan), {a: [] for a in apts}
    lines = ["| fold | rows | AP1 RMSE (s) | fit min |", "|---|---|---|---|"]
    for fold in CV_FOLDS:
        m = tra.month.isin(fold).values
        ix = np.flatnonzero(m)
        tag, t0, r = f"{CACHE}/{fold[0]:02d}{fold[1]:02d}", time.time(), {}
        if os.path.exists(f"{tag}.parquet") and os.path.exists(f"{tag}.json"):
            c = pd.read_parquet(f"{tag}.parquet")
            if np.array_equal(c.id.values, tra.id.values[ix]):
                out[ix], r = c.pred.values, json.load(open(f"{tag}.json"))
        if not r:
            for a in apts:
                ia = (tra.apt == a).values
                models = stages_cb(tra[ia & ~m], va=tra[ia & m], cols=COLS)
                out[ia & m] = predict_cb(models, tra[ia & m], COLS)[0]
                r[a] = [b.tree_count_ for b in models]
            pd.DataFrame({"id": tra.id.values[ix], "pred": out[ix]}).to_parquet(f"{tag}.parquet", index=False)
            json.dump(r, open(f"{tag}.json", "w"), indent=1)
        for a in apts:
            trees[a].append(r[a])
        lines.append(f"| {fold} | {len(ix):,} | {rmse(out[ix] - tra.y.values[ix]):.1f} | {(time.time() - t0) / 60:.0f} |")
        print(lines[-1], flush=True)
    pd.DataFrame({"id": tra.id.values, "pred": out}).to_parquet(OOF_CACHE, index=False)
    return out, {a: [int(x) for x in np.mean(trees[a], axis=0)] for a in apts}, lines


def holdout(tra: pd.DataFrame, tea: pd.DataFrame, rounds: dict[str, list[int]]) -> np.ndarray:
    """SB3 is a five-seed bag, so the challenger is bagged the same way; each airport keeps its own recorded trees."""
    if os.path.exists(TEA_CACHE):
        c = pd.read_parquet(TEA_CACHE)
        if np.array_equal(c.id.values, tea.id.values):
            return c.pred.values
    preds = []
    for s in SEEDS:
        p, t0 = np.full(len(tea), np.nan), time.time()
        for a, r in rounds.items():
            ia, ib = (tra.apt == a).values, (tea.apt == a).values
            models = stages_cb(tra[ia], iters=tuple(r), cols=COLS, over={"random_seed": s})
            p[ib] = predict_cb(models, tea[ib], COLS)[0]
        print(f"seed {s}: {time.time() - t0:.0f} s{score(p, tea)}", flush=True)
        preds.append(p)
    out = np.mean(preds, axis=0)
    pd.DataFrame({"id": tea.id.values, "pred": out}).to_parquet(TEA_CACHE, index=False)
    return out


def main(out: str = "docs/AP1.md") -> None:
    tra, tea = lane_a()
    t0 = time.time()
    pred, rounds, fold_lines = oof(tra)
    ok = np.isfinite(pred)
    cv = rmse(pred[ok] - tra.y.values[ok])

    per = []
    for a, r in rounds.items():
        ia = ok & (tra.apt == a).values
        per.append(f"| {a} | {int(ia.sum()):,} | {rmse(pred[ia] - tra.y.values[ia]):.1f} | {r} |")

    lines = ["# AP1: one CatBoost per airport against the pooled model, Jan + Jul 2025 development holdout", "",
             f"Regenerate with `python -m taxi.ap1`. Declared in protocol section 4 before the code ran; k = 1, one cell, no "
             f"search. D2's cell and recipe unchanged (`taxi.mc1.CELL`, the M2 three-stage split at 600 s, early stopping on the "
             f"fold); the only change is that ten models are fitted, one per airport, on {len(COLS)} columns, D2's 52 less the "
             f"airport itself, which is constant inside every fit. The base is D2's own five-fold CV of {BASE_CV} s (`docs/D2.md`), "
             f"already measured and so the reference rather than a trial. Wall {(time.time() - t0) / 60:.1f} min.", "",
             "## Month-pair CV, pooled over the ten airports", "", *fold_lines, "",
             f"CV RMSE over {int(ok.sum()):,} out-of-fold rows: **{cv:.1f} s** against {BASE}'s {BASE_CV} s "
             f"({cv - BASE_CV:+.1f}).", "",
             "## By airport", "", "| airport | out-of-fold rows | AP1 RMSE (s) | trees (clf, tail, body) |", "|---|---|---|---|",
             *per, ""]

    if cv >= BASE_CV:
        lines += [f"AP1 does not beat {BASE}'s {BASE_CV} s on CV, so it never meets the holdout and SB3 stands. Reading: the pooled "
                  f"model was not paying a real price for reading the airport as a categorical. Ten fits see a tenth of the rows "
                  f"each and lose more to that than they win from a free per-airport structure."]
    else:
        y = tea.y.values
        e0, e1 = sb3_predict(tra, tea) - y, floored(holdout(tra, tea, rounds)) - y
        v = paired_delta(e0, e1, tea.day.values)
        vt, jan, jul = is_tail(tea), tea.month.values == 1, tea.month.values == 7
        row = lambda n, e: f"| {n} | {rmse(e):.2f} | {rmse(e[jan]):.1f} | {rmse(e[jul]):.1f} | {rmse(e[vt]):.1f} | {rmse(e[~vt]):.1f} |"  # noqa: E731
        lines += [f"AP1 beats {BASE} on CV by {BASE_CV - cv:.1f} s, so it meets the holdout. SB3 is a five-seed bag and "
                  f"`docs/SEEDS.md` prices the bag at 1.07 s over one seed, so AP1 is bagged the same way at each airport's own "
                  f"recorded trees.", "",
                  f"## Holdout verdict vs SB3 ({len(tea):,} rows)", "",
                  "| model | RMSE | Jan | Jul | tail rows | body rows |", "|---|---|---|---|---|---|",
                  row("SB3", e0), row("AP1 bag", e1), "", verdict_line("AP1", "SB3", v, k=1)]

    text = "\n".join(lines) + "\n"
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
