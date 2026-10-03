"""Isolated monthly inputs and paired competition metrics for contender runs."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from data import DATA, PLAN_STAMPS, _read
from features import CAT, build_frame, loo_mean


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def outer_masks(f: pd.DataFrame, months: tuple[int, ...]) -> tuple[pd.Series, pd.Series]:
    file_month = f["file"].str.slice(14, 16).astype(int)
    return ~file_month.isin((1, 7, *months)), file_month.isin(months)


def masked_frame(raw: pd.DataFrame) -> pd.DataFrame:
    hidden = raw.copy()
    dep = hidden["PHASE_mvt"] == "DEP"
    hidden.loc[dep, "TAXITIME_SEC_mvt"] = np.nan
    hidden.loc[dep, "BLOCK_TIME_UTC_mvt"] = pd.NaT
    hidden["has_plan"] = hidden[PLAN_STAMPS].notna().any(axis=1)
    hidden["month"] = hidden["MVT_TIME_UTC_mvt"].dt.month
    hidden["split"] = "train"
    return build_frame(hidden)


def cold_inputs(out: Path) -> tuple[pd.DataFrame, list[dict]]:
    frames, audit = [], []
    for path in sorted(DATA.glob("training_*.parquet")):
        raw = _read(str(path))
        dep = raw[raw["PHASE_mvt"] == "DEP"]
        if not dep["MVT_ID_mvt"].is_unique:
            raise ValueError(f"duplicate departure IDs in {path.name}")
        truth = dep.set_index("MVT_ID_mvt")["TAXITIME_SEC_mvt"].astype(float)
        f = masked_frame(raw)
        f["y"] = f["id"].map(truth)
        f["file"] = path.name
        frames.append(f)
        audit.append({"file": path.name, "sha256": file_hash(path), "movements": len(raw),
                      "departures": len(dep), "arrivals": int((raw["PHASE_mvt"] == "ARR").sum()),
                      "first": str(raw["MVT_TIME_UTC_mvt"].min()), "last": str(raw["MVT_TIME_UTC_mvt"].max()),
                      "missing_y": int(truth.isna().sum()), "negative_y": int((truth < 0).sum()),
                      "over_12h_y": int((truth > 43200).sum())})
        print(f"cold input {path.name}: {len(f)} departures", flush=True)
    f = pd.concat(frames, ignore_index=True)
    for col in CAT:
        f[col] = f[col].astype("category")
    if not f["id"].is_unique or not np.isfinite(f["y"]).all():
        raise ValueError("training IDs or targets failed the cold input audit")
    out.mkdir(parents=True, exist_ok=True)
    f.to_parquet(out / "cold_features.parquet")
    return f, audit


def prediction_windows(f: pd.DataFrame, predictions: pd.Series) -> pd.DataFrame:
    if not predictions.index.is_unique or not f["id"].isin(predictions.index).all():
        raise ValueError("neighbour predictions need one finite value per departure ID")
    p = f["id"].map(predictions).to_numpy(float)
    if not np.isfinite(p).all():
        raise ValueError("nonfinite neighbour prediction")
    c = np.maximum(p, 0) - f["z_aobt"].values
    ok = (f["has_plan"].values == 1) & np.isfinite(c)
    c = np.where(ok, np.clip(c, -1800, 1800), 0.0)
    return pd.concat([loo_mean(f, ("file", "apt", "day"), 3600, c, ok, "pw"),
                      loo_mean(f, ("file", "apt", "day", "runway"), 3600, c, ok, "pr")[["pr_m"]],
                      loo_mean(f, ("file", "apt", "day"), 1800, c, ok, "pw30")[["pw30_m"]]], axis=1)


def paired_metrics(rows: pd.DataFrame, p0: np.ndarray, p1: np.ndarray) -> dict:
    y = rows["y"].to_numpy(float)
    e0, e1 = (y - p0) ** 2, (y - p1) ** 2
    delta = e0 - e1
    d = pd.DataFrame({"day": rows["day"].values, "month": rows["month"].values,
                      "e0": e0, "e1": e1, "n": 1})
    g = d.groupby(["month", "day"]).sum()
    rng = np.random.default_rng(0)
    draws, blocks = [], []
    for _ in range(2000):
        ix = rng.integers(len(g), size=len(g))
        b = g.iloc[ix].sum()
        draws.append(np.sqrt(b.e0 / b.n) - np.sqrt(b.e1 / b.n))
        sampled = []
        for _, m in g.groupby(level=0, sort=True):
            starts = rng.integers(len(m), size=int(np.ceil(len(m) / 5)))
            ix = ((starts[:, None] + np.arange(5)) % len(m)).ravel()[:len(m)]
            sampled.append(m.iloc[ix].sum())
        b = pd.DataFrame(sampled).sum()
        blocks.append(np.sqrt(b.e0 / b.n) - np.sqrt(b.e1 / b.n))
    day_gain = g.e0 - g.e1
    daily = pd.Series(day_gain.values, index=pd.to_datetime(day_gain.index.get_level_values("day")))
    episode = daily.reindex(pd.date_range(daily.index.min(), daily.index.max()), fill_value=0).rolling(5, min_periods=1).sum().max()
    return {"n": len(y), "control_rmse": float(np.sqrt(e0.mean())), "candidate_rmse": float(np.sqrt(e1.mean())),
            "gain": float(np.sqrt(e0.mean()) - np.sqrt(e1.mean())), "sse_gain": float(delta.sum()),
            "day_ci": np.percentile(draws, [2.5, 97.5]).tolist(),
            "moving5_ci": np.percentile(blocks, [2.5, 97.5]).tolist(),
            "top_gain": {str(n): float(np.sort(delta)[-n:].sum()) for n in (1, 10, 50)},
            "leave_best_day_sse_gain": float(delta.sum() - day_gain.max()),
            "leave_best_5day_episode_sse_gain": float(delta.sum() - episode),
            "by_month": {str(m): {"n": int(v.n.sum()), "gain": float(np.sqrt(v.e0.sum() / v.n.sum()) - np.sqrt(v.e1.sum() / v.n.sum())),
                                   "sse_gain": float(v.e0.sum() - v.e1.sum())} for m, v in d.groupby("month")}}
