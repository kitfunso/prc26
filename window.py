"""Rolling vs expanding training window for one lane A config, judged on a later month pair.
`python window.py ra_cb_wz 11,12 26 99` - window lengths in weeks, the longest arm is the baseline.
Every training row precedes the evaluation pair; early stopping uses that pair, as lanea.cv does."""
from __future__ import annotations

import sys
import time

from data import rmse
from features import PW, build, with_pw
from lanea import CONFIGS, predict
from train import boot_delta

WEEK = 7 * 86400.0


def run(name: str, months: tuple[int, ...], weeks: tuple[int, ...]) -> None:
    cfg, f = CONFIGS[name], build()
    if set(PW) & set(cfg["num"]):
        f = with_pw(f)
    f = f[(f["split"] == "train") & (f["has_plan"] == 1)]
    te = f[f["month"].isin(months)]
    cut = te["t_off"].min()
    pre = f[f["t_off"] < cut]
    p, t0 = {}, time.time()
    for w in sorted(weeks):
        tr = pre[pre["t_off"] >= cut - abs(w) * WEEK]
        if w < 0:
            tr = pre.sample(n=len(tr), random_state=0)
        p[w] = predict(cfg, tr, te, te)[0]
        tag = f"rand-n-of-{abs(w)}w" if w < 0 else f"{w}w"
        print(f"{name} m={months} {tag}: n_train={len(tr)} rmse={rmse(te['y'].values, p[w]):.2f} {time.time() - t0:.0f}s", flush=True)
    base = max(weeks)
    for w in sorted(weeks):
        if w != base:
            d = boot_delta(te["y"].values, p[base], p[w], te["day"].values)
            tag = f"rand-n-of-{abs(w)}w" if w < 0 else f"{w}w"
            print(f"{name} m={months}: {tag} vs {base}w {d[0]:+.2f} [{d[1]:+.2f}, {d[2]:+.2f}]", flush=True)


if __name__ == "__main__":
    run(sys.argv[1], tuple(int(x) for x in sys.argv[2].split(",")), tuple(int(x) for x in sys.argv[3:]))
