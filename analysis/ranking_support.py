"""Cold ranking support and matched, calendar-blocked year-separation audit."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from campaign import file_hash, masked_frame
from data import DATA, _read
from features import ARR, CAT, DAY, NUM, OCC, WX, WZ
from weather import RAW, STATIONS, load_metar

OUT = DATA / "runs" / "contender-20261003"
COLS = list(dict.fromkeys(CAT + NUM + DAY + OCC + WX + ARR + WZ))


def support(train: pd.DataFrame, rank: pd.DataFrame) -> dict:
    categories = {c: {"unseen_rows": int((~rank[c].isin(train[c].dropna()) & rank[c].notna()).sum()),
                      "missing_rows": int(rank[c].isna().sum())} for c in CAT}
    numbers = {}
    for c in (c for c in COLS if c not in CAT):
        a, b = train[c], rank[c]
        lo, hi = float(a.min()), float(a.max())
        numbers[c] = {"train_missing": int(a.isna().sum()), "rank_missing": int(b.isna().sum()),
                      "train_min": lo, "train_max": hi, "rank_min": float(b.min()), "rank_max": float(b.max()),
                      "outside_train_range": int(((b < lo) | (b > hi)).sum()),
                      "train_p99": float(a.quantile(.99)), "rank_p99": float(b.quantile(.99))}
    return {"train_rows": len(train), "rank_rows": len(rank), "categories": categories, "numbers": numbers}


def year_separation(train: pd.DataFrame, rank: pd.DataFrame) -> dict:
    combined = pd.concat([train.assign(year_label=0), rank.assign(year_label=1)], ignore_index=True)
    rng, parts = np.random.default_rng(0), []
    for _, g in combined.groupby(["apt", "month"], observed=True):
        a, b = g[g.year_label == 0], g[g.year_label == 1]
        n = min(len(a), len(b), 3000)
        if n:
            parts.extend([a.iloc[rng.choice(len(a), n, replace=False)], b.iloc[rng.choice(len(b), n, replace=False)]])
    sample = pd.concat(parts, ignore_index=True)
    day = pd.to_datetime(sample.day)
    sample["calendar_block"] = day.dt.strftime("%Y-%m") + ":" + ((day.dt.day - 1) // 5).astype(str)
    sample["group"] = (day.dt.day - 1) // 5 % 5
    for c in CAT:
        sample[c] = sample[c].astype("category")
    y, p, scopes, importance = sample.year_label.values, np.full(len(sample), np.nan), [], np.zeros(len(COLS))
    for group in range(5):
        test = sample.group == group
        fit_blocks, test_blocks = set(sample.loc[~test, "calendar_block"]), set(sample.loc[test, "calendar_block"])
        if fit_blocks & test_blocks:
            raise ValueError("year-separation fit and scored calendar blocks overlap")
        model = lgb.LGBMClassifier(n_estimators=300, learning_rate=.05, num_leaves=63,
                                  n_jobs=12, verbose=-1, random_state=0)
        model.fit(sample.loc[~test, COLS], y[~test], categorical_feature=CAT)
        p[test] = model.predict_proba(sample.loc[test, COLS])[:, 1]
        importance += model.booster_.feature_importance("gain")
        scopes.append({"group": group, "fit_rows": int((~test).sum()), "test_rows": int(test.sum()),
                       "fit_blocks": sorted(fit_blocks), "test_blocks": sorted(test_blocks)})
        print("year separation group", group, flush=True)
    if not np.isfinite(p).all():
        raise ValueError("year-separation predictions do not cover the matched sample")
    sample[["id", "day", "apt", "month", "year_label", "group"]].assign(p=p).to_parquet(OUT / "year-separation.parquet")
    return {"rows": len(sample), "auc": float(roc_auc_score(y, p)),
            "by_month": {int(m): float(roc_auc_score(y[ix], p[ix])) for m, ix in sample.groupby("month").indices.items()},
            "scopes": scopes, "importance": dict(pd.Series(importance / importance.sum(), index=COLS).sort_values(ascending=False))}


def weather_summary(f: pd.DataFrame) -> dict:
    return {"rows": len(f), "snow": int((f.w_sn > 0).sum()), "freezing": int((f.w_fz > 0).sum()),
            "low_visibility": int((f.w_vsby <= 1).sum()), "thunderstorm": int((f.w_ts > 0).sum()),
            "missing_observation": int(f.w_age.isna().sum()), "observation_older_1h": int((f.w_age > 3600).sum()),
            "snow_reports_24h_max": float(f.w_sn24h.max()), "snow_reports_24h_ge15": int((f.w_sn24h >= 15).sum())}


def main() -> None:
    start = time.monotonic()
    train = pd.read_parquet(OUT / "cold_features.parquet")
    raw = _read(str(DATA / "ranking.parquet"))
    rank = masked_frame(raw)
    rank["file"] = "ranking.parquet"
    dep = raw[raw.PHASE_mvt == "DEP"]
    if not rank.id.is_unique or set(rank.id) != set(dep.MVT_ID_mvt) or not dep.TAXITIME_SEC_mvt.isna().all():
        raise ValueError("ranking IDs or hidden target contract failed")
    if ((rank.has_plan == 1) & ~np.isfinite(rank.z_aobt)).any():
        raise ValueError("ranking Lane A lacks a finite AOBT anchor")
    rank.to_parquet(OUT / "cold_ranking.parquet")
    matched = train[train["file"].str.slice(14, 16).astype(int).isin((1, 7))]
    metar = load_metar()
    record = {"ranking_hash": file_hash(DATA / "ranking.parquet"), "feature_hash": file_hash(OUT / "cold_ranking.parquet"),
              "weather_hashes": {st: file_hash(RAW / f"{st}.csv") for st in STATIONS},
              "weather_spans": {str(a): {"first_utc": str(pd.to_datetime(g.t.min(), unit="s", utc=True)),
                                         "last_utc": str(pd.to_datetime(g.t.max(), unit="s", utc=True)), "rows": len(g)}
                                for a, g in metar.groupby("apt", observed=True)},
              "all_training": support(train, rank), "matched_training": support(matched, rank),
              "weather": {"all_training": weather_summary(train), "matched_training": weather_summary(matched),
                          "ranking": weather_summary(rank)}, "year_separation": year_separation(matched, rank),
              "limit": "Feature separation is not evidence that density weighting improves taxi predictions.",
              "runtime_seconds": time.monotonic() - start}
    (OUT / "ranking_support.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({"auc": record["year_separation"]["auc"], "runtime_seconds": record["runtime_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
