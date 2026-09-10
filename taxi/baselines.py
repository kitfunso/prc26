"""Baselines B0-B4, the admission-cap sweep and the champion ladder per lane (protocol section 4)."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from taxi.data import CV_FOLDS, HOLDOUT_MONTHS, TRAIN_MONTHS, load
from taxi.verdict import FLOOR, paired_delta, promote_floor, rmse

CAPS = (3600.0, 7200.0, 14400.0, None)
Model = Callable[[pd.DataFrame, pd.DataFrame, str], np.ndarray]


def _cap(cap: float | None) -> str:
    return "none" if cap is None else str(int(cap))


def admit(tr: pd.DataFrame, lane: str, cap: float | None) -> pd.DataFrame:
    if cap is None:
        return tr
    keep = np.abs(tr.c) <= cap if lane == "A" else (tr.y > 0) & (tr.y <= cap)
    return tr[keep]


def b0(tr: pd.DataFrame, te: pd.DataFrame, lane: str) -> np.ndarray:
    return np.full(len(te), tr.y.mean())


def b1(tr: pd.DataFrame, te: pd.DataFrame, lane: str) -> np.ndarray:
    return te.apt.map(tr.groupby("apt").y.mean()).fillna(tr.y.mean()).values


def b2(tr: pd.DataFrame, te: pd.DataFrame, lane: str) -> np.ndarray:
    key = ["apt", "hour", "wday"]
    cell = tr.groupby(key).y.mean().rename("p").reset_index()
    p = te[key].merge(cell, on=key, how="left").p.values
    return np.where(np.isnan(p), b1(tr, te, lane), p)


def gap(col: str) -> Model:
    def fit_predict(tr: pd.DataFrame, te: pd.DataFrame, lane: str) -> np.ndarray:
        cal = (tr.y - tr[col]).groupby(tr.apt).median()
        return (te[col] + te.apt.map(cal)).values
    return fit_predict


def b4(tr: pd.DataFrame, te: pd.DataFrame, lane: str, cols: list[str] | None = None) -> np.ndarray:
    cols = cols or (["z_anchor", "z_eobt", "z_lobt", "queue"] if lane == "A" else ["z_sched", "queue"])
    out = b1(tr, te, lane)
    for apt, ix in te.groupby("apt").indices.items():
        t = tr[tr.apt == apt]
        if len(t) <= len(cols) + 1:
            continue
        beta = np.linalg.lstsq(np.column_stack([np.ones(len(t)), t[cols].values]), t.y.values, rcond=None)[0]
        out[ix] = np.column_stack([np.ones(len(ix)), te[cols].values[ix]]) @ beta
    return out


LADDER: dict[str, list[tuple[str, Model]]] = {
    "A": [("B0", b0), ("B1", b1), ("B2", b2), ("B3a", gap("z_aobt")), ("B3b", gap("z_eobt")),
          ("B3c", gap("z_lobt")), ("B3d", gap("z_anchor")), ("B4", b4)],
    "B": [("B0", b0), ("B1", b1), ("B2", b2), ("B3s", gap("z_sched")), ("B4", b4)],
}
SWEEP_MODEL: dict[str, Model] = {"A": b4, "B": b1}


def cap_sweep(tr: pd.DataFrame, lane: str) -> tuple[float | None, list[str]]:
    """Month-pair CV of the lane's sweep model under each cap; validation folds are never trimmed."""
    lines = [f"| lane {lane} cap | CV RMSE (s) |", "|---|---|"]
    scores: dict[float | None, float] = {}
    for cap in CAPS:
        sq, n = 0.0, 0
        for fold in CV_FOLDS:
            va = tr[tr.month.isin(fold)].reset_index(drop=True)
            err = SWEEP_MODEL[lane](admit(tr[~tr.month.isin(fold)], lane, cap), va, lane) - va.y.values
            sq += float(np.sum(err * err))
            n += len(err)
        scores[cap] = float(np.sqrt(sq / n))
        lines.append(f"| {_cap(cap)} | {scores[cap]:.1f} |")
    close = [c for c in CAPS if scores[c] <= min(scores.values()) + FLOOR]
    best = None if None in close else max(close)
    lines.append(f"\nChosen cap, lane {lane}: {_cap(best)} (least trimming among caps within {FLOOR:.0f} s of the best CV RMSE).")
    return best, lines


def ladder(tr: pd.DataFrame, te: pd.DataFrame, lane: str, cap: float | None) -> tuple[str, dict[str, np.ndarray], list[str]]:
    lines = [f"| lane {lane} | RMSE | Jan | Jul | improvement vs champion [95%] | verdict (floor {promote_floor(lane):.0f} s) |",
             "|---|---|---|---|---|---|"]
    fit = admit(tr, lane, cap)
    champ, err_champ, preds = "", np.empty(0), {}
    for name, fn in LADDER[lane]:
        pred = fn(fit, te, lane)
        preds[name] = pred
        err = pred - te.y.values
        jan, jul = rmse(err[te.month.values == 1]), rmse(err[te.month.values == 7])
        if not champ:
            lines.append(f"| {name} | {rmse(err):.1f} | {jan:.1f} | {jul:.1f} | first rung | champion |")
            champ, err_champ = name, err
            continue
        v = paired_delta(err_champ, err, te.day.values, lane=lane)
        tag = "WIN, promoted" if v.win else "tie or lose"
        lines.append(f"| {name} | {v.challenger:.1f} | {jan:.1f} | {jul:.1f} | {champ}: {v.delta:+.1f} s [{v.lo:+.1f}, {v.hi:+.1f}] | {tag} |")
        if v.win:
            champ, err_champ = name, err
    return champ, preds, lines


def main() -> None:
    tr, te = load(TRAIN_MONTHS), load(HOLDOUT_MONTHS)
    report = ["# Baselines B0-B4 on the Jan+Jul 2025 development holdout", "",
              f"Train months {TRAIN_MONTHS}; regenerate with `python -m taxi.baselines`.", "",
              "| rows | training | holdout |", "|---|---|---|",
              f"| lane A (flight plan matched) | {int(tr.has_fpl.sum())} | {int(te.has_fpl.sum())} |",
              f"| lane B (no flight plan) | {int((~tr.has_fpl).sum())} | {int((~te.has_fpl).sum())} |", ""]
    lane_te, lane_pred, champions = {}, {}, {}
    for lane in ("B", "A"):
        trl = tr[tr.has_fpl == (lane == "A")].reset_index(drop=True)
        tel = te[te.has_fpl == (lane == "A")].reset_index(drop=True)
        cap, lines = cap_sweep(trl, lane)
        report += [f"## Lane {lane}: admission-cap sweep", ""] + lines + [""]
        champ, preds, lines = ladder(trl, tel, lane, cap)
        report += [f"## Lane {lane}: champion ladder (cap {_cap(cap)})", ""] + lines + [""]
        lane_te[lane], lane_pred[lane], champions[lane] = tel, preds, champ
    err_b = lane_pred["B"][champions["B"]] - lane_te["B"].y.values
    report += [f"## All rows (reported, never decides): lane-A rung plus lane-B champion {champions['B']}", "",
               "| rung | all-row RMSE |", "|---|---|"]
    for name, pred in lane_pred["A"].items():
        report.append(f"| {name} | {rmse(np.concatenate([pred - lane_te['A'].y.values, err_b])):.1f} |")
    report += ["", f"Champions: lane A {champions['A']}, lane B {champions['B']}."]
    text = "\n".join(report) + "\n"
    with open("docs/BASELINES.md", "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
