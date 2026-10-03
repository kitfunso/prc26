"""Training-only blend weights for a surviving neural loss comparison."""
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
from features import ARR, CAT, DAY, NUM, OCC, WX, WZ
from neighbours import with_scoped_pw
from train import fit_predict, stopping_split

OUT = DATA / "runs" / "contender-20261003"


def simplex_weight(y: np.ndarray, base: np.ndarray, extra: np.ndarray) -> float:
    d = extra - base
    return float(np.clip(np.dot(y - base, d) / max(np.dot(d, d), 1e-12), 0, 1))


def meta_inputs(f: pd.DataFrame, months: tuple[int, ...], full: bool, target: Path, block: dict) -> tuple:
    fit, eva = outer_masks(f, months)
    fit &= f.has_plan == 1
    training = fit.copy()
    if not full:
        training &= (pd.to_datetime(f.day) - pd.Timestamp("2025-01-01")).dt.days % 5 == 0
    tr = f[training]
    inner, stop = stopping_split(tr)
    subset = f[outer_masks(f, months)[0] | eva].copy().reset_index(drop=True)
    if full:
        tag = "-".join(map(str, months))
        pw = pd.read_parquet(target / f"pw-inner-{tag}.parquet")
        if not pw.id.is_unique or set(pw.id) != set(subset.id):
            raise ValueError("cached inner PW IDs differ from the declared full-capacity scope")
        inside = subset.merge(pw, on="id", validate="one_to_one")
        scopes = block["inner_donor_scopes"]
    else:
        inside, scopes = with_scoped_pw(subset, pd.Index(inner.id))
    if set(stop.day.unique()) != set(block["inner_stop_days"]):
        raise ValueError("meta stopping dates differ from neural budget-selection dates")
    return fit, inner, stop, inside, scopes


def run_block(f: pd.DataFrame, block: dict, full: bool, target: Path) -> dict:
    months = tuple(block["months"])
    tag = "-".join(map(str, months))
    fit, inner, stop, inside, scopes = meta_inputs(f, months, full, target, block)
    data = torch.load(target / f"CP3-mse-{tag}.pt", map_location="cpu", weights_only=False)
    cfg = data["config"]
    selected = inside[inside.id.isin(inner.id)]
    model, enc, mu, sd, _ = nn.train(cfg, selected, epochs=block["models"]["CP3-mse"]["epochs"])
    stop_f = inside[inside.id.isin(stop.id)]
    ci, xn = enc(stop_f)
    p_nn = np.maximum(nn._infer(model, ci, xn) * sd + mu + stop_f.z_aobt.values, 0)
    del model, enc, ci, xn, data
    torch.cuda.empty_cache()
    fit_tree = f[fit & ~f.day.isin(stop.day.unique())]
    tree = {"family": "lgb", "cat": CAT, "num": NUM + DAY + OCC + WX + ARR + WZ,
            "params": {"objective": "regression", "learning_rate": .08, "num_leaves": 63,
                       "min_child_samples": 100, "n_estimators": 350, "n_jobs": 12,
                       "verbose": -1, "random_state": 0}}
    p_tree, _ = fit_predict(tree, fit_tree.assign(y=fit_tree.y - fit_tree.z_aobt), stop_f, iters=350)
    p_tree = np.maximum(p_tree + stop_f.z_aobt.values, 0)
    w = simplex_weight(stop_f.y.values, p_tree, p_nn)
    control = pd.read_parquet(OUT / f"screen-{tag}.parquet")
    neural_rows = pd.read_parquet(target / f"neural-{tag}.parquet").set_index("id")
    if not neural_rows.index.is_unique or set(control.id) != set(neural_rows.index):
        raise ValueError("meta members do not cover identical complete evaluation files")
    extra = control.id.map(neural_rows["CP3-mse"])
    if not np.isfinite(extra).all() or (extra < 0).any():
        raise ValueError("meta neural member has invalid predictions")
    if not np.array_equal(extra[control.has_plan == 0], control.control[control.has_plan == 0]):
        raise ValueError("meta neural member changed the frozen Lane B control")
    weights = {"established": .388830, "simplex": w, "shrunk": .5 * (w + .388830)}
    record = {"months": months, "weights": weights, "meta_fit_rows": len(stop_f),
              "meta_fit_days": stop_f.day.unique().tolist(), "tree_fit_rows": len(fit_tree),
              "nn_fit_rows": len(selected), "donor_scopes": scopes, "results": {}}
    for name, weight in weights.items():
        p = (1 - weight) * control.control.values + weight * extra.values
        control[name] = p
        record["results"][name] = paired_metrics(control, control.control.values, p)
        hot = (control.w_ts > 0) | (control.w_sn > 0) | (control.w_fz > 0) | (control.w_vsby <= 1)
        record["results"][name]["hot"] = paired_metrics(control[hot], control.control.values[hot], p[hot])
        print(tag, name, json.dumps(record["results"][name]), flush=True)
    control.to_parquet(target / f"meta-{tag}.parquet")
    return record


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    full = sys.argv[1:2] == ["full"]
    target = OUT / "neural-full" if full else OUT
    neural = json.loads((target / "neural.json").read_text(encoding="utf-8"))
    if len(neural["blocks"]) != 2 or any(b["mse_vs_huber"]["gain"] <= 0 for b in neural["blocks"]):
        raise RuntimeError("MSE needs a consistent two-block loss advantage before this meta screen")
    t0 = time.monotonic()
    f = pd.read_parquet(OUT / "cold_features.parquet")
    for c in CAT:
        f[c] = f[c].astype("category")
    records = []
    for block in neural["blocks"]:
        records.append(run_block(f, block, full, target))
        (target / "meta.json").write_text(json.dumps({"blocks": records, "runtime_seconds": time.monotonic() - t0,
                                                     "capacity": "all permitted fit days" if full else "every fifth fit day"}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
