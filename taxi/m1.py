"""M1: LightGBM per lane on the masked feature frame; grid by month-pair CV; verdict vs B4 (protocol section 4)."""
from __future__ import annotations

import re
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from taxi.baselines import b4
from taxi.data import CATEGORICAL, CV_FOLDS, FEATURES, HOLDOUT_MONTHS, PLAN_CAT, TRAIN_MONTHS, load
from taxi.verdict import FLOOR, paired_delta, rmse

MODEL_FEATURES = ["apt", *FEATURES]
CAT = ["apt", *CATEGORICAL, *PLAN_CAT]  # every categorical the view carries; matrix() and the fits keep only those in cols
LABEL = {"A": "c", "B": "y"}
GRID = {
    "A": [dict(num_leaves=n, min_data_in_leaf=m, objective=o)
          for n in (31, 127, 255) for m in (200, 2000) for o in ("l2", "huber")],
    "B": [dict(num_leaves=n, min_data_in_leaf=m, objective=o)
          for n in (7, 31) for m in (50, 200) for o in ("l2", "huber")],
}
BASE = dict(learning_rate=0.1, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, alpha=600.0,
            metric="l2", verbose=-1, seed=0)
MAX_ROUNDS, PATIENCE = 2000, 50


def matrix(df: pd.DataFrame, cols: list[str] = MODEL_FEATURES) -> pd.DataFrame:
    x = df[cols].copy()
    for c in CAT:
        if c in cols:
            x[c] = x[c].astype("category")
    return x


def train(fit: pd.DataFrame, lane: str, cell: dict, va: pd.DataFrame | None = None, rounds: int = MAX_ROUNDS,
          cols: list[str] = MODEL_FEATURES) -> lgb.Booster:
    params = {**BASE, **cell}
    cats = [c for c in CAT if c in cols]
    dtrain = lgb.Dataset(matrix(fit, cols), fit[LABEL[lane]].values, categorical_feature=cats)
    if va is None:
        return lgb.train(params, dtrain, num_boost_round=rounds)
    dval = lgb.Dataset(matrix(va, cols), va[LABEL[lane]].values, reference=dtrain)
    return lgb.train(params, dtrain, num_boost_round=rounds, valid_sets=[dval],
                     callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])


def predict(bst: lgb.Booster, df: pd.DataFrame, lane: str) -> np.ndarray:
    raw = bst.predict(matrix(df, bst.feature_name()), num_iteration=bst.best_iteration or None)
    return raw + df.z_aobt.values if lane == "A" else raw


def cv_rmse(tr: pd.DataFrame, lane: str, cell: dict, rounds: int, cols: list[str]) -> float:
    sq, n = 0.0, 0
    for fold in CV_FOLDS:
        m = tr.month.isin(fold).values
        err = predict(train(tr[~m], lane, cell, rounds=rounds, cols=cols), tr[m], lane) - tr.y.values[m]
        sq, n = sq + float(err @ err), n + len(err)
    return float(np.sqrt(sq / n))


def knockout(tr: pd.DataFrame, lane: str, cell: dict, rounds: int) -> list[str]:
    """Drop one feature at a time at the chosen cell and fixed rounds; positive loss means the feature earns its place."""
    full = cv_rmse(tr, lane, cell, rounds, MODEL_FEATURES)
    lines = [f"Lane {lane}, full model CV RMSE {full:.1f} s at {cell}, {rounds} rounds.", "",
             "| dropped feature | CV RMSE (s) | loss vs full (s) | clears floor |", "|---|---|---|---|"]
    for f in MODEL_FEATURES:
        score = cv_rmse(tr, lane, cell, rounds, [c for c in MODEL_FEATURES if c != f])
        lines.append(f"| {f} | {score:.1f} | {score - full:+.1f} | {'yes' if score - full >= FLOOR else 'no'} |")
        print(lines[-1], flush=True)
    return lines


def chosen_cells(path: str = "docs/M1.md") -> dict[str, tuple[dict, int]]:
    """Read the chosen cell and rounds per lane back from the M1 report."""
    import ast
    import re
    text = open(path, encoding="utf-8").read()
    out = {}
    for lane in ("A", "B"):
        section = text.split(f"## Lane {lane}: grid")[1]
        m = re.search(r"Chosen cell \d+: (\{.*?\}), (\d+) rounds", section)
        out[lane] = (ast.literal_eval(m.group(1)), int(m.group(2)))
    return out


def main_knockout() -> None:
    tr = load(TRAIN_MONTHS)
    report = ["# M1 per-feature knockout (protocol section 0, complexity control)", "",
              "Month-pair CV at the chosen cell and rounds, one feature dropped at a time; regenerate with `python -m taxi.m1 knockout`.", ""]
    for lane, (cell, rounds) in chosen_cells().items():
        trl = tr[tr.has_fpl == (lane == "A")].reset_index(drop=True)
        report += [f"## Lane {lane}", ""] + knockout(trl, lane, cell, rounds) + [""]
    text = "\n".join(report) + "\n"
    with open("docs/M1-KNOCKOUT.md", "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


def main_sensitivity() -> None:
    """Critique finding 3: the lane-A verdict without the negative-AOBT-gap rows; a diagnostic that never decides."""
    tr, te = load(TRAIN_MONTHS), load(HOLDOUT_MONTHS)
    tra, tea = tr[tr.has_fpl].reset_index(drop=True), te[te.has_fpl].reset_index(drop=True)
    cell, rounds = chosen_cells()["A"]
    e_m1 = predict(train(tra, "A", cell, rounds=rounds), tea, "A") - tea.y.values
    e_b4 = b4(tra, tea, "A") - tea.y.values
    neg = (tea.z_aobt < 0).values
    print(f"lane A holdout rows {len(tea)}, z_aobt < 0 on {int(neg.sum())} rows in {tea.day[neg].nunique()} days; "
          f"RMSE on those rows B4 {rmse(e_b4[neg]):.1f}, M1 {rmse(e_m1[neg]):.1f}", flush=True)
    for name, m in (("all rows", np.ones(len(tea), bool)), ("z_aobt >= 0 rows", ~neg)):
        v = paired_delta(e_b4[m], e_m1[m], tea.day.values[m])
        print(f"{name}: B4 {v.champion:.1f}, M1 {v.challenger:.1f}, {v.line()}", flush=True)


def final_rounds(name: str, path: str) -> tuple[int, int, int]:
    """The M2-recipe rounds (LightGBM) or trees (CatBoost) a lane report recorded, e.g. final_rounds("W1", "docs/W1.md")."""
    m = re.search(rf"final {name} (?:rounds|trees) \((\d+), (\d+), (\d+)\)", open(path, encoding="utf-8").read())
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def expected_max(k: int) -> float:
    """E[max of k standard normals], the optimism bound multiplier for a k-cell search."""
    return float(np.random.default_rng(0).standard_normal((200_000, k)).max(axis=1).mean())


def cv(tr: pd.DataFrame, lane: str) -> tuple[dict, int, list[str]]:
    """Month-pair CV of every grid cell; the chosen cell is the most parsimonious within the floor of the best."""
    lines = ["| cell | leaves | min_data | objective | CV RMSE (s) | rounds | per-fold rounds | fit s |", "|---|---|---|---|---|---|---|---|"]
    results = []
    for i, cell in enumerate(GRID[lane], 1):
        t0, sq, n, rounds = time.time(), 0.0, 0, []
        for fold in CV_FOLDS:
            m = tr.month.isin(fold).values
            bst = train(tr[~m], lane, cell, va=tr[m])
            err = predict(bst, tr[m], lane) - tr.y.values[m]
            sq, n = sq + float(err @ err), n + len(err)
            rounds.append(bst.best_iteration)
        score, r = float(np.sqrt(sq / n)), int(np.mean(rounds))
        results.append((score, cell, r, i))
        lines.append(f"| {i} | {cell['num_leaves']} | {cell['min_data_in_leaf']} | {cell['objective']} | {score:.1f} | {r} | {rounds} | {time.time() - t0:.0f} |")
        print(lines[-1], flush=True)
    best = min(s for s, *_ in results)
    close = [x for x in results if x[0] <= best + FLOOR]
    score, cell, r, i = min(close, key=lambda x: (x[1]["num_leaves"], -x[1]["min_data_in_leaf"]))
    lines.append(f"\nChosen cell {i}: {cell}, {r} rounds, CV RMSE {score:.1f} s (best {best:.1f}; most parsimonious within {FLOOR:.0f} s).")
    return cell, r, lines


def importance(bst: lgb.Booster) -> list[str]:
    gain = bst.feature_importance("gain")
    share = gain / gain.sum()
    order = np.argsort(-share)
    return ["| feature | gain share |", "|---|---|"] + [f"| {bst.feature_name()[j]} | {share[j]:.3f} |" for j in order]


def main() -> None:
    tr, te = load(TRAIN_MONTHS), load(HOLDOUT_MONTHS)
    report = ["# M1: LightGBM per lane on the Jan+Jul 2025 development holdout", "",
              f"Train months {TRAIN_MONTHS}; regenerate with `python -m taxi.m1`. Features: {MODEL_FEATURES}.", "",
              f"Lane A learns c = y - z_aobt, lane B learns y. Base params {BASE}; early stopping {PATIENCE} rounds on the validation fold, max {MAX_ROUNDS}.", ""]
    chosen, lane_te, lane_err = {}, {}, {}
    for lane in ("B", "A"):
        trl = tr[tr.has_fpl == (lane == "A")].reset_index(drop=True)
        tel = te[te.has_fpl == (lane == "A")].reset_index(drop=True)
        cell, rounds, lines = cv(trl, lane)
        chosen[lane] = (cell, rounds)
        k = len(GRID[lane])
        report += [f"## Lane {lane}: grid (k = {k}) by month-pair CV", ""] + lines + [""]
        bst = train(trl, lane, cell, rounds=rounds)
        p_m1, p_b4 = predict(bst, tel, lane), b4(trl, tel, lane)
        e_m1, e_b4 = p_m1 - tel.y.values, p_b4 - tel.y.values
        v = paired_delta(e_b4, e_m1, tel.day.values)
        bound = expected_max(k) * (v.hi - v.lo) / 3.92
        clears = v.win and v.delta >= bound + FLOOR
        jan, jul = tel.month.values == 1, tel.month.values == 7
        report += [f"## Lane {lane}: holdout verdict vs champion B4 ({len(tel)} rows)", "",
                   "| model | RMSE | Jan | Jul |", "|---|---|---|---|",
                   f"| B4 | {rmse(e_b4):.1f} | {rmse(e_b4[jan]):.1f} | {rmse(e_b4[jul]):.1f} |",
                   f"| M1 | {rmse(e_m1):.1f} | {rmse(e_m1[jan]):.1f} | {rmse(e_m1[jul]):.1f} |", "",
                   f"Improvement M1 vs B4: {v.delta:+.1f} s [{v.lo:+.1f}, {v.hi:+.1f}]; paired SE {(v.hi - v.lo) / 3.92:.1f} s; "
                   f"optimism bound for k = {k}: {bound:.1f} s; floor {FLOOR:.0f} s. "
                   f"Verdict: {'WIN, clears bound plus floor, M1 promoted' if clears else ('WIN on the interval but inside bound plus floor, not promoted' if v.win else 'tie or lose')}.", "",
                   f"### Lane {lane}: feature gain share (final fit)", ""] + importance(bst) + [""]
        lane_te[lane], lane_err[lane] = tel, (e_m1 if clears else e_b4, e_m1, e_b4)
    champ_err = np.concatenate([lane_err["A"][0], lane_err["B"][0]])
    both_m1 = np.concatenate([lane_err["A"][1], lane_err["B"][1]])
    both_b4 = np.concatenate([lane_err["A"][2], lane_err["B"][2]])
    report += ["## All rows (reported, never decides)", "", "| combination | all-row RMSE |", "|---|---|",
               f"| B4 + B4 | {rmse(both_b4):.1f} |", f"| M1 + M1 | {rmse(both_m1):.1f} |",
               f"| lane champions after this verdict | {rmse(champ_err):.1f} |", ""]
    fwd, dec = load(tuple(range(1, 12))), load((12,))
    report += ["## Forward fold (reported, never decides): train Jan-Nov 2025, predict Dec 2025", "",
               "| lane | rows | B4 RMSE | M1 RMSE |", "|---|---|---|---|"]
    for lane in ("A", "B"):
        f = fwd[fwd.has_fpl == (lane == "A")].reset_index(drop=True)
        d = dec[dec.has_fpl == (lane == "A")].reset_index(drop=True)
        cell, rounds = chosen[lane]
        e_m1 = predict(train(f, lane, cell, rounds=rounds), d, lane) - d.y.values
        e_b4 = b4(f, d, lane) - d.y.values
        report.append(f"| {lane} | {len(d)} | {rmse(e_b4):.1f} | {rmse(e_m1):.1f} |")
    text = "\n".join(report) + "\n"
    with open("docs/M1.md", "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    import sys
    {"knockout": main_knockout, "sensitivity": main_sensitivity}.get(sys.argv[1] if sys.argv[1:] else "", main)()
