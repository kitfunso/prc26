"""Leakage and drift audit: adversarial 2025-vs-2026 classifier, feature drift table, lane-B tail shares. Writes docs/AUDIT.md."""
from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from features import CAT, NUM, build


def adversarial(f: pd.DataFrame, cols: list[str]) -> tuple[float, pd.Series]:
    """AUC of train-vs-rank on the given columns; 0.5 means no drift, features ranked by gain."""
    x = f[cols]
    y = (f["split"] == "rank").astype(int).values
    rng = np.random.default_rng(0)
    m = rng.random(len(f)) < 0.7
    clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63, verbose=-1, random_state=0)
    clf.fit(x[m], y[m], categorical_feature=[c for c in cols if c in CAT])
    auc = roc_auc_score(y[~m], clf.predict_proba(x[~m])[:, 1])
    imp = pd.Series(clf.booster_.feature_importance("gain"), index=cols).sort_values(ascending=False)
    return float(auc), imp / imp.sum()


def drift_table(f: pd.DataFrame) -> pd.DataFrame:
    """Quantiles of every numeric feature on train Jan+Jul vs rank (same months a year on)."""
    a = f[(f["split"] == "train") & f["month"].isin((1, 7))]
    b = f[f["split"] == "rank"]
    rows = []
    for c in NUM:
        rows.append({"feat": c, "na_tr%": 100 * a[c].isna().mean(), "na_rk%": 100 * b[c].isna().mean(),
                     "p50_tr": a[c].median(), "p50_rk": b[c].median(), "p90_tr": a[c].quantile(.9), "p90_rk": b[c].quantile(.9)})
    return pd.DataFrame(rows).set_index("feat").round(1)


def tail(f: pd.DataFrame) -> str:
    """Where the squared error lives in the training label, by lane and band."""
    t = f[f["split"] == "train"]
    out = []
    for lane, g in t.groupby(t["has_plan"].map({1: "A", 0: "B"})):
        band = pd.cut(g["y"], [-1, 600, 1200, 1800, 3600, 7200, 21600, 72000, 1e9])
        mu = g["y"].mean()
        sse = (g["y"] - mu) ** 2
        tab = pd.DataFrame({"n": band.value_counts(sort=False), "share_n%": 100 * band.value_counts(sort=False, normalize=True),
                            "sse%": 100 * sse.groupby(band, observed=False).sum() / sse.sum()}).round(2)
        out += [f"\n### lane {lane}, n={len(g):,}, mean {mu:.0f}, rmse(mean) {np.sqrt(sse.mean()):.0f}\n", tab.to_markdown()]
    return "\n".join(out)


def main() -> None:
    f = build()
    lines = ["# Leakage and drift audit\n"]
    auc_all, imp = adversarial(f, CAT + NUM)
    lines += [f"\n## Adversarial validation (train 2025 vs rank 2026), all features: AUC {auc_all:.4f}\n", imp.head(12).round(3).to_markdown()]
    no_time = [c for c in CAT + NUM if c not in ("stand", "runway", "operator", "ades", "airline", "actype")]
    auc_nt, imp2 = adversarial(f, no_time)
    lines += [f"\n\nwithout stand/runway/operator/ades/airline/actype: AUC {auc_nt:.4f}\n", imp2.head(8).round(3).to_markdown()]
    lines += ["\n\n## Feature quantiles, train Jan+Jul 2025 vs rank 2026\n", drift_table(f).to_markdown()]
    lines += ["\n\n## Where the label variance sits (train)\n", tail(f)]
    Path("docs/AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
