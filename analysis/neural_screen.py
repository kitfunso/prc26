"""Fixed Huber/MSE comparison with nested donor sources and complete refits."""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import nn
from campaign import outer_masks, paired_metrics
from data import DATA
from features import ARR, CAT, DAY, NUM, OCC, PW, WX, WZ
from neighbours import with_scoped_pw
from train import stopping_split

OUT = DATA / "runs" / "contender-20261003"
SCREEN = ((2, 3), (9, 10))


def fit_one(cfg: dict, train_inner: pd.DataFrame, train_full: pd.DataFrame, te: pd.DataFrame, path: Path) -> tuple[np.ndarray, dict]:
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    selected = nn.train(cfg, train_inner)
    selection_info, ep = selected[0].run_info.copy(), selected[4]
    selected_seconds = time.monotonic() - start
    del selected
    model, enc, mu, sd, _ = nn.train(cfg, train_full, epochs=ep)
    if model.run_info["fit_rows"] != len(train_full) or model.run_info["fit_days"] != train_full.day.nunique():
        raise ValueError("full refit omitted permitted training rows or days")
    ci, xn = enc(te)
    p = np.maximum(nn._infer(model, ci, xn) * sd + mu + te.z_aobt.values, 0)
    torch.save({"state_dict": model.state_dict(), "encoder": enc, "mu": mu, "sd": sd,
                "config": cfg, "run_info": model.run_info}, path)
    info = {"selection": selection_info, "refit": model.run_info, "epochs": ep,
            "selection_seconds": selected_seconds, "refit_seconds": time.monotonic() - start - selected_seconds,
            "peak_gpu_bytes": torch.cuda.max_memory_allocated()}
    del model, enc, ci, xn
    torch.cuda.empty_cache()
    return p, info


def run_block(f: pd.DataFrame, months: tuple[int, ...], full: bool, target: Path) -> dict:
    tag = "-".join(map(str, months))
    fit, eva = outer_masks(f, months)
    fit &= f.has_plan == 1
    if not full:
        fit &= (pd.to_datetime(f.day) - pd.Timestamp("2025-01-01")).dt.days % 5 == 0
    training = f[fit]
    inner, stop = stopping_split(training)
    subset = f[outer_masks(f, months)[0] | eva].copy().reset_index(drop=True)
    inside, inner_scopes = with_scoped_pw(subset, pd.Index(inner.id))
    complete, outer_scopes = with_scoped_pw(subset, pd.Index(training.id))
    train_inner = inside[inside.id.isin(training.id)]
    train_full = complete[complete.id.isin(training.id)]
    te = complete[complete.id.isin(f.loc[eva, "id"]) & (complete.has_plan == 1)]
    rows = pd.read_parquet(OUT / f"screen-{tag}.parquet")
    if set(te.id) != set(rows.loc[rows.has_plan == 1, "id"]):
        raise ValueError("neural evaluation IDs do not cover the complete Lane A file")
    record = {"months": months, "fit_rows": len(training), "fit_days": int(training.day.nunique()),
              "inner_stop_days": stop.day.unique().tolist(), "inner_donor_scopes": inner_scopes,
              "outer_donor_scopes": outer_scopes, "models": {}, "results": {}}
    for label, frame in (("inner", inside), ("outer", complete)):
        frame[["id", *PW]].to_parquet(target / f"pw-{label}-{tag}.parquet")
    for loss in ("huber", "mse"):
        name = "CP3-" + loss
        cfg = {"cell": {**nn.CELL_T7, "max_epochs": 150 if full else 30, "loss": loss, "progress": True},
               "cat": CAT, "num": NUM + DAY + OCC + WX + ARR + WZ + PW}
        p, record["models"][name] = fit_one(cfg, train_inner, train_full, te, target / f"{name}-{tag}.pt")
        rows[name] = rows.control
        rows.loc[rows.has_plan == 1, name] = rows.loc[rows.has_plan == 1, "id"].map(pd.Series(p, index=te.id))
        if not np.isfinite(rows[name]).all() or (rows[name] < 0).any():
            raise ValueError("assembled evaluation file has invalid predictions")
        record["results"][name] = paired_metrics(rows, rows.control.values, rows[name].values)
        print(tag, name, json.dumps(record["results"][name]), flush=True)
        rows.to_parquet(target / f"neural-{tag}.parquet")
        (target / f"progress-{tag}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    record["mse_vs_huber"] = paired_metrics(rows, rows["CP3-huber"].values, rows["CP3-mse"].values)
    return record


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if nn.DEV != "cuda":
        raise RuntimeError("this declared neural screen requires the available CUDA GPU")
    t0 = time.monotonic()
    full = sys.argv[1:2] == ["full"]
    target = OUT / "neural-full" if full else OUT
    target.mkdir(exist_ok=True)
    f = pd.read_parquet(OUT / "cold_features.parquet")
    for col in CAT:
        f[col] = f[col].astype("category")
    records = []
    for months in SCREEN:
        record = run_block(f, months, full, target)
        records.append(record)
        limit = "All permitted fit days; two reused development blocks, not complete CV." if full else "One-fifth of fit days; 30-epoch screen cap. Not full-data confirmation."
        (target / "neural.json").write_text(json.dumps({"blocks": records, "runtime_seconds": time.monotonic() - t0,
                                                       "limit": limit}, indent=2), encoding="utf-8")
        comparison = record["mse_vs_huber"]
        fit_seconds = sum(m["selection_seconds"] + m["refit_seconds"] for b in records for m in b["models"].values())
        reversal = comparison["day_ci"][1] < 0 and comparison["moving5_ci"][1] < 0
        if full and (reversal or fit_seconds >= 4 * 3600):
            reason = "clear full-capacity MSE reversal" if reversal else "declared four-hour GPU-fit budget reached"
            (target / "decision.json").write_text(json.dumps({"status": "stopped", "reason": reason,
                                                              "completed_blocks": len(records), "fit_seconds": fit_seconds,
                                                              "loss_comparison": comparison}, indent=2), encoding="utf-8")
            logging.info("Full comparison stopped: %s", reason)
            break


if __name__ == "__main__":
    main()
