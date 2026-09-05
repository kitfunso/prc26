"""W1 weather lane (protocol section 4): lane A = M2 recipe plus the METAR columns; lane B = B4 plus five weather regressors."""
from __future__ import annotations

import re
import time

import numpy as np
import pandas as pd

from taxi.baselines import b1, b4
from taxi.data import CV_FOLDS, HOLDOUT_MONTHS, TRAIN_MONTHS, load
from taxi.m1 import MODEL_FEATURES, chosen_cells, expected_max
from taxi.m2 import K, is_tail, predict_m2, stages
from taxi.verdict import FLOOR, Verdict, paired_delta, rmse
from taxi.weather import PULL, WEATHER, join, load_metar

COLS_A = [*MODEL_FEATURES, *WEATHER]
WB = ["w_sknt", "w_vsby", "w_p3h", "w_ts3h", "w_sn"]


def m2_rounds(path: str = "docs/M2.md") -> tuple[int, int, int]:
    m = re.search(r"final M2 rounds \(classifier, tail, body\) \((\d+), (\d+), (\d+)\)", open(path, encoding="utf-8").read())
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def b4w(tr: pd.DataFrame, te: pd.DataFrame) -> np.ndarray:
    """B4 lane-B regression with the weather columns added; gaps take the airport's training median."""
    cols = ["z_sched", "queue", *WB]
    out = b1(tr, te, "B")
    for apt, ix in te.groupby("apt").indices.items():
        t = tr[tr.apt == apt]
        if len(t) <= len(cols) + 1:
            continue
        med = t[cols].median().fillna(0.0)
        x_tr, x_te = t[cols].fillna(med).values, te[cols].iloc[ix].fillna(med).values
        beta = np.linalg.lstsq(np.column_stack([np.ones(len(t)), x_tr]), t.y.values, rcond=None)[0]
        out[ix] = np.column_stack([np.ones(len(ix)), x_te]) @ beta
    return out


def cv_a(tr: pd.DataFrame, cell: dict, cols: list[str] = COLS_A, name: str = "W1") -> tuple[tuple[int, int, int], list[str]]:
    lines = [f"| fold | {name} RMSE (s) | {name} rounds (clf, tail, body) | fit s |", "|---|---|---|---|"]
    sq, n, rounds = 0.0, 0, []
    for fold in CV_FOLDS:
        m, t0 = tr.month.isin(fold).values, time.time()
        va = tr[m]
        models = stages(tr[~m], cell, va=va, cols=cols)
        e = predict_m2(models, va)[0] - va.y.values
        sq, n = sq + float(e @ e), n + len(e)
        rounds.append([b.best_iteration for b in models])
        lines.append(f"| {fold} | {rmse(e):.1f} | {rounds[-1]} | {time.time() - t0:.0f} |")
        print(lines[-1], flush=True)
    r = tuple(int(x) for x in np.mean(rounds, axis=0))
    lines.append(f"\nCV RMSE over the five folds: {name} {np.sqrt(sq / n):.1f} s (M2 on the same rows: see docs/M2.md); final {name} rounds {r}.")
    print(lines[-1], flush=True)
    return r, lines


def verdict_line(name: str, champ: str, v: Verdict, k: int = K) -> str:
    """Promotion wording only when the comparator is a current registry champion (critique S3, finding 2)."""
    from taxi.champ import CURRENT
    bound = expected_max(k) * (v.hi - v.lo) / 3.92
    clears = v.win and v.delta >= bound + FLOOR
    word = (f"WIN, clears bound plus floor, {name} promoted" if champ in CURRENT.values() else
            f"WIN, clears bound plus floor against {champ}, NOT promoted: the current champions are {CURRENT}") if clears else (
        f"WIN on the interval but inside bound plus floor, {champ} stays" if v.win else f"tie or lose, {champ} stays")
    return (f"Improvement {name} vs {champ}: {v.delta:+.1f} s [{v.lo:+.1f}, {v.hi:+.1f}]; paired SE {(v.hi - v.lo) / 3.92:.1f} s; "
            f"optimism bound for k = {k}: {bound:.1f} s; floor {FLOOR:.0f} s. Verdict: {word}.")


def coverage(df: pd.DataFrame, name: str) -> str:
    miss = df.w_sknt.isna().mean()
    return f"{name}: {len(df)} rows, no observation before the anchor on {100 * miss:.2f}%; observation age median {np.nanmedian(df.w_age) / 60:.0f} min, p95 {np.nanpercentile(df.w_age, 95) / 60:.0f} min."


def main() -> None:
    metar = load_metar()
    tr, te = (join(load(m), metar) for m in (TRAIN_MONTHS, HOLDOUT_MONTHS))
    cov = [coverage(tr, "train"), coverage(te, "holdout")]
    print("\n".join(cov), flush=True)
    tra, tea = (d[d.has_fpl].reset_index(drop=True) for d in (tr, te))
    trb, teb = (d[~d.has_fpl].reset_index(drop=True) for d in (tr, te))
    # Lane B first: seconds of work, and its verdict does not depend on lane A.
    eb0, eb1 = b4(trb, teb, "B") - teb.y.values, b4w(trb, teb) - teb.y.values
    vb = paired_delta(eb0, eb1, teb.day.values)
    jan_b, jul_b = teb.month.values == 1, teb.month.values == 7
    rowb = lambda name, e: f"| {name} | {rmse(e):.1f} | {rmse(e[jan_b]):.1f} | {rmse(e[jul_b]):.1f} | {e.min():.0f} | {e.max():.0f} |"
    lane_b = [f"## Lane B holdout verdict vs champion B4 ({len(teb)} rows)", "", "| model | RMSE | Jan | Jul | min error | max error |",
              "|---|---|---|---|---|---|", rowb("B4", eb0), rowb("B4 + weather", eb1), "", verdict_line("W1-B", "B4", vb), ""]
    print("\n".join(lane_b), flush=True)
    cell, _ = chosen_cells()["A"]
    rounds, lines = cv_a(tra, cell)
    p1 = predict_m2(stages(tra, cell, rounds=rounds, cols=COLS_A), tea)[0]
    p2 = predict_m2(stages(tra, cell, rounds=m2_rounds()), tea)[0]
    e1, e2 = p1 - tea.y.values, p2 - tea.y.values
    va = paired_delta(e2, e1, tea.day.values)
    vt, jan, jul = is_tail(tea), tea.month.values == 1, tea.month.values == 7
    row = lambda name, e: f"| {name} | {rmse(e):.1f} | {rmse(e[jan]):.1f} | {rmse(e[jul]):.1f} | {rmse(e[vt]):.1f} | {rmse(e[~vt]):.1f} |"
    report = ["# W1: weather lane, Jan+Jul 2025 development holdout", "",
              f"Train months {TRAIN_MONTHS}; regenerate with `python -m taxi.w1`. METAR pull `data/weather/{PULL}` (IEM ASOS archive, "
              f"hashes in its manifest.json). Weather columns joined at the anchor time (latest observation at or before it): {list(WEATHER)}. "
              f"Lane A = M2 recipe (cell {cell}) on {len(COLS_A)} columns, k = {K}; lane B = B4 plus {WB}, k = {K}.", "",
              "## Coverage", "", *cov, "", "## Lane A month-pair CV", ""] + lines + ["",
              f"## Lane A holdout verdict vs champion M2 ({len(tea)} rows; tail rows {int(vt.sum())})", "",
              "| model | RMSE | Jan | Jul | tail rows | body rows |", "|---|---|---|---|---|---|", row("M2", e2), row("W1-A", e1), "",
              verdict_line("W1-A", "M2", va), ""] + lane_b
    text = "\n".join(report) + "\n"
    with open("docs/W1.md", "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
