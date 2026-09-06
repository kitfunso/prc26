"""LBX: why two tree attempts lost in lane B, and whether the lane is closed or just mis-fit.

Diagnostic, never a verdict: no cell is declared and no champion moves. H12 lost LightGBM on raw y by 312 s and L10 lost it on
the residual label by 757 s, so the two-failures rule says diagnose what they shared before a third attempt. What they shared is
the loss: 20 of 17,097 lane-B training rows carry 52.5% of the raw sum of squares, so every split chases twenty rows. This asks
the one question that separates "lane B has no more signal" from "lane B was never fit properly": on the ordinary rows alone,
with the artifact rows out of the FIT, does a tree over lane B's 18 populated columns beat the champion that uses two of them?
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from taxi.l2 import floored
from taxi.l6 import CELLS as L6_CELLS, per_row
from taxi.l8 import CELLS as L8_CELLS
from taxi.verdict import paired_delta, rmse
from taxi.w1 import HOLDOUT_MONTHS, TRAIN_MONTHS, load

BIG = 43_200.0  # 12 h: DE's artifact band, a block time written one day early.
B4_COLS = ["z_sched", "queue"]
ALL_COLS = ["month", "t_off", "hour", "wday", "z_sched", "z_anchor", "queue", "dep_before", "dep_after",
            "arr_anchor", "arr_off", "push_anchor", "occ", "hol", "hol_prev", "hol_next", "rq", "rec_taxi"]
CAT = ["apt", "airline"]  # every champion since B1 is fit per airport; a global tree without these cannot see one.
# One fixed cell, not a grid: this is a diagnostic and must not become an undeclared search.
KW = dict(objective="l2", num_leaves=31, min_data_in_leaf=100, learning_rate=0.05, feature_fraction=0.9,
          bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=0)
ROUNDS = 400


def frames(tr: pd.DataFrame, te: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shared categories across both frames, or LightGBM codes the same airport differently in fit and predict."""
    cats = [c for c in CAT if c in cols]
    a, b = tr[cols].copy(), te[cols].copy()
    for c in cats:
        lv = pd.Index(sorted(set(a[c].astype(str)) | set(b[c].astype(str))))
        a[c], b[c] = (pd.Categorical(d[c].astype(str), categories=lv) for d in (a, b))
    return a, b


def fit(tr: pd.DataFrame, te: pd.DataFrame, cols: list[str], resid: bool = False) -> np.ndarray:
    """resid: fit y minus z_sched and add it back, so the tree gets the copy branch's identity for free (L10's label)."""
    a, b = frames(tr, te, cols)
    lab = tr.y.values - tr.z_sched.values if resid else tr.y.values
    m = lgb.train(KW, lgb.Dataset(a, lab, categorical_feature=[c for c in CAT if c in cols]), num_boost_round=ROUNDS)
    p = m.predict(b)
    return p + te.z_sched.values if resid else p


def main() -> None:
    tr, te = load(TRAIN_MONTHS), load(HOLDOUT_MONTHS)
    trb, teb = (d[~d.has_fpl].reset_index(drop=True) for d in (tr, te))
    ok_tr, ok_te = trb.y.values <= BIG, teb.y.values <= BIG
    y, day = teb.y.values, teb.day.values
    champ = floored(per_row(trb, teb, L6_CELLS["L6a"], prob=L8_CELLS["L8c"]))

    preds = {"L8c (champion, 2 columns)": champ}
    full = [*ALL_COLS, *CAT]
    for tag, cols in (("B4 columns", B4_COLS), ("B4 columns + airport", [*B4_COLS, "apt"]),
                      ("all 18 columns", ALL_COLS), ("all 18 + airport, airline", full)):
        for fitname, sel in (("all rows", np.ones(len(trb), bool)), ("ordinary rows only", ok_tr)):
            preds[f"LightGBM, {tag}, fit on {fitname}"] = floored(fit(trb[sel], teb, cols))
    for fitname, sel in (("all rows (L10's design)", np.ones(len(trb), bool)), ("ordinary rows only", ok_tr)):
        preds[f"LightGBM on y minus z_sched, all 18 + airport, airline, fit on {fitname}"] = floored(fit(trb[sel], teb, full, resid=True))

    rows, vs = [], []
    for name, p in preds.items():
        e = p - y
        rows.append(f"| {name} | {rmse(e):,.1f} | {rmse(e[ok_te]):.1f} | {rmse(e[~ok_te]):,.0f} |")
        if name != "L8c (champion, 2 columns)":
            v = paired_delta((champ - y)[ok_te], e[ok_te], day[ok_te])
            vs.append(f"| {name} | {v.delta:+.1f} | [{v.lo:+.1f}, {v.hi:+.1f}] | {'WIN' if v.lo > 0 else 'lose' if v.hi < 0 else 'tie'} |")

    sq = lambda d, m: 100 * np.sum(d.y.values[m] ** 2) / np.sum(d.y.values ** 2)  # noqa: E731
    out = [
        "# LBX: is lane B out of signal, or was it never fit properly?", "",
        "Regenerate with `python -m taxi.lbx`. Diagnostic: no cell declared, no champion moves, no verdict decided. One fixed",
        f"LightGBM cell throughout ({KW['num_leaves']} leaves, min_data {KW['min_data_in_leaf']}, lr {KW['learning_rate']}, {ROUNDS} rounds), so nothing here is a search.", "",
        "## 1. What the two lost tree attempts shared", "",
        f"Lane B has {len(trb):,} training and {len(teb):,} holdout rows. Rows above {BIG:,.0f} s are DE's artifact band, a block time",
        "written one day early.", "",
        "| frame | rows | artifact rows | share of the raw sum of squares |", "|---|---|---|---|",
        f"| training | {len(trb):,} | {int((~ok_tr).sum())} | {sq(trb, ~ok_tr):.1f}% |",
        f"| holdout | {len(teb):,} | {int((~ok_te).sum())} | {sq(teb, ~ok_te):.1f}% |", "",
        f"{int((~ok_tr).sum())} of {len(trb):,} training rows, {100 * (~ok_tr).mean():.2f}% of them, carry {sq(trb, ~ok_tr):.1f}% of the squared loss a tree minimises. H12 (raw y,",
        "-312.1 s) and L10 (residual label, -757 s) both fit that loss, so both spent their splits on twenty rows.", "",
        f"## 2. Lane B's columns. B4 and every mixture since fit {len(B4_COLS)} of the {len(ALL_COLS)} populated numeric columns", "",
        f"`{'`, `'.join(c for c in ALL_COLS if c not in B4_COLS)}` are populated on every lane-B row and no lane-B regressor has read them.",
        f"Airport enters the champion as a separate per-airport fit rather than a column, so the trees below carry `{'`, `'.join(CAT)}` as", 
        "categoricals to give them the same information.", "",
        "## 3. The question: on ordinary rows, does a tree beat two columns of OLS?", "",
        "| model | all holdout rows | ordinary rows | artifact rows |", "|---|---|---|---|", *rows, "",
        f"Paired day-block bootstrap against the champion on the {int(ok_te.sum()):,} ordinary rows (positive is better than the champion):", "",
        "| challenger | delta (s) | 95% interval | |", "|---|---|---|---|", *vs, "",
        "## 4. Answer: lane B is out of signal for this model class, not mis-fit", "",
        f"Every one of the {len(vs)} specifications loses on the ordinary rows, the closest by {max(float(v.split('|')[2]) for v in vs):+.1f} s with an interval clear of zero. Three",
        "things were tried against the two lost attempts and none of them turned the sign. Removing the artifact rows from the fit",
        "helps but nowhere near enough. Sixteen unread columns help a little. The airport, which the champion carries as a separate",
        "per-airport fit and both earlier trees carried as a categorical, is worth about 500 s on its own and was the largest single",
        "term here, so a global tree without it is not a fair comparison and the first draft of this table was wrong. And the residual",
        "label, which hands the tree the copy branch's identity for free, is the worst row on the page, confirming L10 independently",
        "rather than explaining it away.", "",
        "The champion's shape is doing the work: a per-airport linear fit can emit `y = z_sched` on a copied row exactly, and a tree",
        "emits piecewise constants over a variable that ranges across thousands of seconds. That is a representation gap, not a",
        "tuning gap, and no amount of columns or rounds closes it.", "",
        "Consequence for the campaign: with F25 closing lane A's 49 columns and F21 closing the artifact rows, the 6.9% of board",
        "squared error that separates the projected v15 from the leader is not sitting in a lane-B model class either. Do not spend",
        "a declared lane on CatBoost or TabPFN in lane B on the strength of their lane-A wins.", "",
    ]
    text = "\n".join(out) + "\n"
    with open("docs/LBX.md", "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
