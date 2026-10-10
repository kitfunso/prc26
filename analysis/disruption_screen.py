"""Two preregistered Lane A feature recipes against the frozen clean CPU screen."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from campaign import outer_masks, paired_metrics
from data import DATA
from disruption import recorder_features, weather_features
from features import ARR, CAT, DAY, NUM, OCC, WX, WZ
from train import fit_predict

OUT = DATA / "runs" / "contender-20261003"
SCREEN = ((2, 3), (9, 10))


def signed_predict(tr: pd.DataFrame, te: pd.DataFrame, cfg: dict) -> np.ndarray:
    y = tr.y.values - tr.z_aobt.values
    lab = np.where(y > 600, 2, np.where(y < -600, 0, 1))
    cols = cfg["cat"] + cfg["num"]
    gate = lgb.LGBMClassifier(learning_rate=.05, num_leaves=31, min_child_samples=100,
                             n_estimators=120, n_jobs=12, random_state=0, verbose=-1)
    gate.fit(tr[cols], lab, categorical_feature=CAT)
    prob = gate.predict_proba(te[cols])
    residual = np.zeros(len(te))
    for j, c in enumerate(gate.classes_):
        rows = tr[lab == c].assign(y=y[lab == c])
        pred, _ = fit_predict(cfg, rows, te, iters=350)
        residual += prob[:, j] * pred
    return np.maximum(residual + te.z_aobt.values, 0)


def main() -> None:
    t0 = time.monotonic()
    f = pd.read_parquet(OUT / "cold_features.parquet")
    for col in CAT:
        f[col] = f[col].astype("category")
    weather, recorder = weather_features(f), recorder_features(f)
    base_num = NUM + DAY + OCC + WX + ARR + WZ
    groups = {"CP2-weather": list(weather), "CP2-recorder": list(recorder)}
    f = pd.concat([f, weather, recorder], axis=1)
    prior_path = OUT / "disruption.json"
    results = json.loads(prior_path.read_text(encoding="utf-8"))["blocks"] if prior_path.exists() else []
    blocks = SCREEN if len(sys.argv) == 1 else (tuple(int(x) for x in sys.argv[1].split("-")),)
    for months in blocks:
        tag = "-".join(map(str, months))
        fit, eva = outer_masks(f, months)
        fit &= f.has_plan == 1
        eva &= f.has_plan == 1
        tr, te = f[fit], f[eva]
        all_rows = pd.read_parquet(OUT / f"screen-{tag}.parquet")
        result = {"months": months, "fit_rows": len(tr), "results": {}}
        for name, extra in groups.items():
            if months == (9, 10) and name == "CP2-recorder" and results[0]["results"][name]["gain"] < 0:
                result["results"][name] = {"verdict": "Stopped after first-block whole-file and hot-slice harm"}
                continue
            cfg = {"family": "lgb", "cat": CAT, "num": base_num + extra,
                   "params": {"objective": "regression", "learning_rate": .08, "num_leaves": 63,
                              "min_child_samples": 100, "n_estimators": 350, "n_jobs": 12,
                              "verbose": -1, "random_state": 0}}
            if name == "CP2-recorder":
                p = signed_predict(tr, te, cfg)
            else:
                p, _ = fit_predict(cfg, tr.assign(y=tr.y - tr.z_aobt), te, iters=350)
                p = np.maximum(p + te.z_aobt.values, 0)
            all_rows[name] = all_rows.control
            all_rows.loc[all_rows.has_plan == 1, name] = all_rows.loc[all_rows.has_plan == 1, "id"].map(pd.Series(p, index=te.id))
            result["results"][name] = paired_metrics(all_rows, all_rows.control.values, all_rows[name].values)
            hot = (all_rows.w_ts > 0) | (all_rows.w_sn > 0) | (all_rows.w_fz > 0) | (all_rows.w_vsby <= 1)
            if hot.any():
                result["results"][name]["hot"] = paired_metrics(all_rows[hot], all_rows.control.values[hot], all_rows[name].values[hot])
            print(tag, name, json.dumps(result["results"][name]), flush=True)
        all_rows.to_parquet(OUT / f"disruption-{tag}.parquet")
        results.append(result)
        (OUT / "disruption.json").write_text(json.dumps({"blocks": results, "runtime_seconds": time.monotonic() - t0}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
