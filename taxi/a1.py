"""A1: the anchor guard (F11). `before` snapshots every champion on the old anchors; `after` re-verdicts each one on the fixed data, one H row each."""
from __future__ import annotations

import os
import shutil
import sys
import time

import numpy as np
import pandas as pd

from taxi import l4, l6, mc1  # noqa: F401  eager: the lazy imports inside champ must not pick up a half-edited module while this job runs
from taxi.champ import CHAMPIONS_A, CURRENT, cache_path, champion_holdout, predict_b, refit_predict
from taxi.data import COUNTS, HOLDOUT_MONTHS, TRAIN_MONTHS, load
from taxi.l2 import floored
from taxi.verdict import paired_delta, rmse
from taxi.w1 import verdict_line
from taxi.weather import join, load_metar

SNAP = ("z_anchor", *COUNTS, "occ", "t_occ")
PRE = "data/cache/anchor.pre-a1.parquet"
L6A_PRE = "data/cache/l6a_holdout.pre-a1.parquet"
NOISE = "data/cache/mc2_holdout.refit2.pre-a1.parquet"


def frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metar = load_metar()
    tr, te = (join(load(m), metar) for m in (TRAIN_MONTHS, HOLDOUT_MONTHS))
    tra, tea, trb, teb = (d[d.has_fpl == f].reset_index(drop=True) for f in (True, False) for d in (tr, te))
    return tra, tea, trb, teb


def before() -> None:
    tra, tea, trb, teb = frames()
    for name in CHAMPIONS_A:
        if not os.path.exists(cache_path(name) + ".pre-a1"):
            shutil.copyfile(cache_path(name), cache_path(name) + ".pre-a1")
    pd.concat([d[["id", *SNAP]] for d in (tra, tea, trb, teb)]).to_parquet(PRE, index=False)
    lb = CURRENT["B"]
    pd.DataFrame({"id": teb.id.values, "pred": predict_b(lb, trb, teb)}).to_parquet(L6A_PRE, index=False)
    print(f"{lb} on the old anchors: floored RMSE {rmse(floored(pd.read_parquet(L6A_PRE).pred.values) - teb.y.values):.1f}", flush=True)
    t0 = time.time()
    p2, p1 = refit_predict("MC2", tra, tea), champion_holdout("MC2", tra, tea)
    pd.DataFrame({"id": tea.id.values, "pred": p2}).to_parquet(NOISE, index=False)
    y = tea.y.values
    v = paired_delta(p1 - y, p2 - y, tea.day.values)
    print(f"MC2 same-seed refit on the old anchors ({time.time() - t0:.0f} s): cache {rmse(p1 - y):.2f}, refit {rmse(p2 - y):.2f}, "
          f"paired delta {v.delta:+.2f} [{v.lo:+.2f}, {v.hi:+.2f}]; that is the GPU refit noise floor for the A1 MC2 row", flush=True)


def after() -> None:
    tra, tea, trb, teb = frames()
    pre = pd.read_parquet(PRE).set_index("id")
    now = pd.concat([d[["id", *SNAP]] for d in (tra, tea, trb, teb)]).set_index("id").loc[pre.index]
    changed = {c: int((pre[c].fillna(-1).values != now[c].fillna(-1).values).sum()) for c in SNAP}
    y, days = tea.y.values, tea.day.values
    noise = pd.read_parquet(NOISE).pred.values - y
    lines = ["| champion | lane | old RMSE (s) | new RMSE (s) | new vs old, paired [95%] | note |", "|---|---|---|---|---|---|"]
    verdicts = []
    for name in CHAMPIONS_A:
        old = pd.read_parquet(cache_path(name) + ".pre-a1")
        assert np.array_equal(old.id.values, tea.id.values)
        e0, e1 = old.pred.values - y, champion_holdout(name, tra, tea) - y
        v = paired_delta(e0, e1, days)
        note = f"GPU refit noise floor on the old anchors: {rmse(noise):.2f} s vs cache {rmse(e0):.2f} s, paired {paired_delta(e0, noise, days).delta:+.2f} s" if name == "MC2" else "LightGBM, deterministic refit"
        lines.append(f"| {name} | A | {rmse(e0):.2f} | {rmse(e1):.2f} | {v.delta:+.2f} [{v.lo:+.2f}, {v.hi:+.2f}] | {note} |")
        verdicts.append(verdict_line(f"{name} on A1 anchors", name, v, k=1))
        print(lines[-1], flush=True)
    lb, yb = CURRENT["B"], teb.y.values
    old_b = pd.read_parquet(L6A_PRE)
    assert np.array_equal(old_b.id.values, teb.id.values)
    e0, e1 = floored(old_b.pred.values) - yb, floored(predict_b(lb, trb, teb)) - yb
    v = paired_delta(e0, e1, teb.day.values, lane="B")
    lines.append(f"| {lb} | B | {rmse(e0):.1f} | {rmse(e1):.1f} | {v.delta:+.1f} [{v.lo:+.1f}, {v.hi:+.1f}] | floored (H22) |")
    verdicts.append(verdict_line(f"{lb} on A1 anchors", lb, v, k=1))
    report = ["# A1: anchor guard, every champion re-verdicted on the fixed data (Jan+Jul 2025 development holdout)", "",
              "Regenerate with `python -m taxi.a1 after` (after `python -m taxi.a1 before` ran on the old anchors). The anchor is now the first non-negative of the AOBT, "
              "EOBT and schedule gaps, else the takeoff itself (z = 0), for every lane; before A1 it was AOBT, else EOBT, unchecked (F11). "
              "Each row is the same champion refit on the fixed data against its own predictions on the old data: a data fix, not a promotion.", "",
              f"Rows changed by the guard over the 12 months of 2025 ({len(pre):,} departures): " + ", ".join(f"{c} {n:,}" for c, n in changed.items()) + ".", "",
              *lines, "", *verdicts, ""]
    text = "\n".join(report)
    with open("docs/A1.md", "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    {"before": before, "after": after}[sys.argv[1]]()
