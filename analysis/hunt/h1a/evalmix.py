"""Q2/Q4 payoff on development months: does the copy detector beat v34 on LIRF rows with zs > 30 min?
Mixture p_mix = pc * zs + (1 - pc) * phys, phys = mean non-copy y (y <= 12 h) of the training folds' rows.
Stack: y - p = a * (zs - p) + b * pc * (zs - p), coefficients fit on the other folds only.
Run: PYTHONPATH=. python analysis/hunt/h1a/evalmix.py  (needs copyprob_lirf.parquet from copymodel.py)"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from common import OUT
from copymodel import FOLDS

HULL = 14400
BIG = 43200


def phys_mean(train: pd.DataFrame) -> float:
    s = train[(train["copy"] == 0) & (train["y"] <= BIG)]
    return float(s["y"].mean())


def stack_coef(train: pd.DataFrame) -> np.ndarray:
    g = train["zs"] - train["p"]
    x = np.c_[g, train["pc"] * g]
    return np.linalg.lstsq(x, (train["y"] - train["p"]).values, rcond=None)[0]


def logit(p: pd.Series) -> pd.Series:
    q = p.clip(0.01, 0.99)
    return np.log(q / (1 - q))


def oof(dv: pd.DataFrame) -> pd.DataFrame:
    """Per-fold phys mean, stack and combined-logistic coefficients from the other dev folds (judgeable rows)."""
    dv = dv.copy()
    for f in FOLDS:
        te_m = dv["month"].isin(f)
        tr = dv[~te_m & (dv["y"] <= BIG) & (dv["zs"] < HULL)]
        ph = phys_mean(tr)
        dv.loc[te_m, "phys"] = ph
        a, b = stack_coef(tr)
        g = dv.loc[te_m, "zs"] - dv.loc[te_m, "p"]
        dv.loc[te_m, "p_stack"] = dv.loc[te_m, "p"] + a * g + b * dv.loc[te_m, "pc"] * g
        dv.loc[te_m, "coef"] = f"a={a:.3f} b={b:.3f}"
        pv = lambda s: ((s["p"] - ph) / (s["zs"] - ph)).clip(0, 1)
        xtr = np.c_[logit(pv(tr)), logit(tr["pc"])]
        lr = LogisticRegression(C=1.0).fit(xtr, tr["copy"].values)
        te_d = dv[te_m]
        pcomb = lr.predict_proba(np.c_[logit(pv(te_d)), logit(te_d["pc"])])[:, 1]
        dv.loc[te_m, "p_comb"] = pcomb * te_d["zs"] + (1 - pcomb) * ph
    dv["p_mix"] = dv["pc"] * dv["zs"] + (1 - dv["pc"]) * dv["phys"]
    dv["pv"] = ((dv["p"] - dv["phys"]) / (dv["zs"] - dv["phys"])).clip(0, 1)
    dv["p_v34mix"] = dv["pv"] * dv["zs"] + (1 - dv["pv"]) * dv["phys"]
    return dv


def report(dv: pd.DataFrame, lane: str) -> None:
    for name, m in (("judgeable (zs < 4 h, y <= 12 h)", (dv["zs"] < HULL) & (dv["y"] <= BIG)),
                    ("all zs > 30 min", dv["y"].notna())):
        s = dv[m]
        e = {c: ((s["y"] - s[c]) ** 2).sum() / 1e9 for c in ("p", "p_mix", "p_stack", "p_comb", "p_v34mix")}
        print(f"lane {lane} {name}: n={len(s)} SSE v34 {e['p']:.3f}B mix {e['p_mix']:.3f}B stack {e['p_stack']:.3f}B"
              f" comb {e['p_comb']:.3f}B v34-as-mixture {e['p_v34mix']:.3f}B")
    s = dv[(dv["zs"] < HULL) & (dv["y"] <= BIG)]
    for zb, t in s.groupby(pd.cut(s["zs"] / 3600, [0.5, 1, 2, 4]), observed=True):
        auc = lambda c: roc_auc_score(t["copy"], t[c]) if t["copy"].nunique() == 2 else np.nan
        d_mix = (((t["y"] - t["p"]) ** 2).sum() - ((t["y"] - t["p_mix"]) ** 2).sum()) / 1e6
        d_st = (((t["y"] - t["p"]) ** 2).sum() - ((t["y"] - t["p_stack"]) ** 2).sum()) / 1e6
        print(f"  zs {zb}: n={len(t)} AUC pc {auc('pc'):.3f} v34-implied {auc('pv'):.3f}"
              f" | SSE saved: mix {d_mix:.1f}M stack {d_st:.1f}M")
    for f in FOLDS:
        t = s[s["month"].isin(f)]
        d_st = (((t["y"] - t["p"]) ** 2).sum() - ((t["y"] - t["p_stack"]) ** 2).sum()) / 1e6
        d_mix = (((t["y"] - t["p"]) ** 2).sum() - ((t["y"] - t["p_mix"]) ** 2).sum()) / 1e6
        print(f"  fold {f}: n={len(t)} saved mix {d_mix:.1f}M stack {d_st:.1f}M ({t['coef'].iloc[0]})")


def main() -> None:
    r = pd.read_parquet(OUT / "copyprob_lirf.parquet")
    for lane in ("B", "A"):
        dv = oof(r[(r["part"] == "dev") & (r["lane"] == lane)])
        report(dv, lane)


if __name__ == "__main__":
    main()
