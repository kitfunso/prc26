"""Declared CP1 screens on cold inputs; never writes the established global caches."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import laneb
from campaign import cold_inputs, file_hash, masked_frame, outer_masks, paired_metrics
from data import DATA, _read
from features import ARR, CAT, DAY, NUM, OCC, WX, WZ
from train import fit_predict

OUT = DATA / "runs" / "contender-20261003"
SCREEN = ((2, 3), (9, 10))


def cold_check() -> dict:
    raw = _read(str(sorted(DATA.glob("training_*.parquet"))[0]))
    first = masked_frame(raw)
    mutated = raw.copy()
    dep = mutated["PHASE_mvt"] == "DEP"
    mutated.loc[dep, "TAXITIME_SEC_mvt"] = -987654.0
    mutated.loc[dep, "BLOCK_TIME_UTC_mvt"] = pd.Timestamp("2040-01-01", tz="UTC")
    second = masked_frame(mutated)
    pd.testing.assert_frame_equal(first, second)
    third = masked_frame(_read(str(sorted(DATA.glob("training_*.parquet"))[0])))
    pd.testing.assert_frame_equal(first, third)
    return {"file": sorted(DATA.glob("training_*.parquet"))[0].name, "rows": len(first),
            "hidden_label_and_block_mutation": "PASS", "independent_cold_build": "PASS"}


def main() -> None:
    t0 = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "inputs.json").exists():
        manifest = json.loads((OUT / "inputs.json").read_text(encoding="utf-8"))
        f = pd.read_parquet(OUT / "cold_features.parquet")
        for col in CAT:
            f[col] = f[col].astype("category")
    else:
        f, audit = cold_inputs(OUT)
        check = cold_check()
        manifest = {"run_id": OUT.name, "cold_audit": audit, "cold_check": check,
                    "feature_hash": file_hash(OUT / "cold_features.parquet"), "blocks": [],
                    "limit": "Reduced-capacity clean CPU control, not a reconstruction of clean-v53."}
        (OUT / "inputs.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("cold build and hidden-label mutation passed", flush=True)
    from mechanisms import candidates
    blocks = SCREEN if len(sys.argv) == 1 else (tuple(int(x) for x in sys.argv[1].split("-")),)
    manifest["blocks"] = []
    for months in blocks:
        tag = "-".join(map(str, months))
        fit, eva = outer_masks(f, months)
        # The same complete outer file and unchanged Lane A predictions close each paired comparison.
        cfg = {"family": "lgb", "cat": CAT, "num": NUM + DAY + OCC + WX + ARR + WZ,
               "params": {"objective": "regression", "learning_rate": .08, "num_leaves": 63,
                          "min_child_samples": 100, "n_estimators": 350, "n_jobs": 12,
                          "verbose": -1, "random_state": 0}}
        ta, ha = f[fit & (f.has_plan == 1)], f[eva & (f.has_plan == 1)]
        tr = ta.assign(y=ta.y - ta.z_aobt)
        existing = OUT / f"screen-{tag}.parquet"
        if existing.exists():
            previous = pd.read_parquet(existing).set_index("id")
            pa = ha.id.map(previous.control).values
        else:
            pa, _ = fit_predict(cfg, tr, ha, iters=350)
            pa = np.maximum(pa + ha.z_aobt.values, 0)
        fb = laneb.augment(f)
        bt, bv, pb, it = laneb.outer_predict(fb, fit, pd.Index(f.loc[eva & (f.has_plan == 0), "id"]))
        pb = laneb.hull(bv, laneb.hull_prior(bt, bv, pb))
        predictions = pd.Series(np.r_[pa, pb], index=np.r_[ha.id.values, bv.id.values])
        rows = f[eva].copy()
        rows["control"] = rows.id.map(predictions)
        variants, detail = candidates(bt, bv, pb)
        from mechanisms import copy_labels
        for prefix in ("raw", "calibrated"):
            probability = detail.pop(prefix + "_probabilities")
            detail[prefix + "_brier"] = float(brier_score_loss(copy_labels(bv), probability))
            detail[prefix + "_logloss"] = float(log_loss(copy_labels(bv), probability, labels=[0, 1]))
        record = {"months": months, "fit_rows": int(fit.sum()), "eval_rows": len(rows),
                  "fit_ids_hash": __import__("hashlib").sha256(f.loc[fit, "id"].to_numpy().tobytes()).hexdigest(),
                  "lanea_iterations": 350, "laneb_iterations": it, "mechanism_detail": detail, "results": {}}
        for name, p in variants.items():
            rows[name] = rows["control"]
            rows.loc[rows.has_plan == 0, name] = rows.loc[rows.has_plan == 0, "id"].map(pd.Series(p, index=bv.id))
            record["results"][name] = paired_metrics(rows, rows.control.values, rows[name].values)
            record["results"][name]["laneB"] = paired_metrics(bv, pb, p)
            print(tag, name, json.dumps(record["results"][name]), flush=True)
        rows.to_parquet(OUT / f"screen-{tag}.parquet")
        manifest["blocks"].append(record)
        manifest["runtime_seconds"] = time.monotonic() - t0
        (OUT / f"screen-{tag}.json").write_text(json.dumps(record, indent=2, default=lambda v: v.item()), encoding="utf-8")
        (OUT / "screen.json").write_text(json.dumps(manifest, indent=2, default=lambda v: v.item()), encoding="utf-8")


if __name__ == "__main__":
    main()
