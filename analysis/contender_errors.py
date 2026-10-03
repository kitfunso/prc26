"""Descriptive error slices and influential rows for recorded contender files."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import laneb
from campaign import file_hash
from data import DATA
from mix import label

OUT = DATA / "runs" / "contender-20261003"


def error_rows(rows: pd.DataFrame, name: str) -> dict:
    if not rows.id.is_unique or not np.isfinite(rows[["control", name, "y"]]).all().all():
        raise ValueError("error analysis needs unique IDs and finite observed labels/predictions")
    d = rows[["id", "day", "month", "apt", "has_plan", "y", "z_aobt", "z_sched"]].copy()
    d["e0"], d["e1"] = (rows.y - rows.control) ** 2, (rows.y - rows[name]) ** 2
    d["sse_gain"] = d.e0 - d.e1
    d["lane"] = np.where(d.has_plan == 1, "A", "B")
    d["hot"] = (rows.w_ts > 0) | (rows.w_sn > 0) | (rows.w_fz > 0) | (rows.w_vsby <= 1)
    states = np.array(["aobt_copy", "eobt_copy", "schedule_copy", "other"])[label(rows)]
    copy, near, _, shift = laneb.classes(rows)
    states[d.has_plan == 0] = np.select([shift, copy, near], ["day_shift", "schedule_copy", "near_copy"], default="other")[d.has_plan == 0]
    d["recording_class"] = states
    d["signed_residual"] = np.select([d.has_plan == 0, d.y - d.z_aobt < -600, d.y - d.z_aobt > 600],
                                      ["no_aobt", "negative_tail", "positive_tail"], default="body")
    d["gap"] = pd.cut(d.z_sched, [-np.inf, 3600, 14400, 57600, np.inf],
                      labels=["under_1h", "1_to_4h", "4_to_16h", "over_16h"])
    slices = {}
    for column in ("lane", "apt", "month", "hot", "recording_class", "signed_residual", "gap"):
        g = d.groupby(column, observed=True).agg(n=("id", "size"), e0=("e0", "sum"), e1=("e1", "sum"))
        g["control_rmse"], g["candidate_rmse"] = np.sqrt(g.e0 / g.n), np.sqrt(g.e1 / g.n)
        g["gain"] = g.control_rmse - g.candidate_rmse
        g["sse_gain"] = g.e0 - g.e1
        slices[column] = {str(k): v for k, v in g.drop(columns=["e0", "e1"]).to_dict("index").items()}
    columns = ["id", "day", "apt", "lane", "y", "z_aobt", "z_sched", "recording_class", "hot", "sse_gain"]
    top = {}
    for direction in ("gain", "harm"):
        selected = d.sort_values("sse_gain", ascending=direction == "harm").head(50)[columns]
        top[direction] = selected.astype(object).where(selected.notna(), None).to_dict("records")
    return {"slices": slices, "influential_rows": top}


def main() -> None:
    families = {"screen": ("CP1-hierarchy", "CP1-calibration", "CP1-outcomes"),
                "disruption": ("CP2-weather", "CP2-recorder"), "neural": ("CP3-huber", "CP3-mse"),
                "meta": ("established", "simplex", "shrunk"),
                "neural-full/neural": ("CP3-huber", "CP3-mse"),
                "neural-full/meta": ("established", "simplex", "shrunk")}
    record = {"limit": "Descriptive, label-based mechanism slices; not features or new promotion rules.", "files": {}}
    for family, names in families.items():
        for tag in ("2-3", "9-10"):
            path = OUT / f"{family}-{tag}.parquet"
            if not path.exists():
                continue
            rows = pd.read_parquet(path)
            record["files"][f"{family}-{tag}"] = {"sha256": file_hash(path), "recipes": {}}
            for name in names:
                if name in rows:
                    record["files"][f"{family}-{tag}"]["recipes"][name] = error_rows(rows, name)
    (OUT / "error_slices.json").write_text(json.dumps(record, indent=2, default=lambda v: v.item(), allow_nan=False), encoding="utf-8")
    print(json.dumps({"files": len(record["files"]), "file_recipe_pairs": sum(len(x["recipes"]) for x in record["files"].values()),
                      "status": "measured descriptive slices"}))


if __name__ == "__main__":
    main()
