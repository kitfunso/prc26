"""Resume the full champion control without using shared label-derived caches."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from campaign import file_hash, outer_masks
from champion import fit_member
from data import DATA
from features import CAT

OUT = DATA / "runs" / "clean-champion-20261003"
INPUT = DATA / "runs" / "contender-20261003"


def inputs() -> pd.DataFrame:
    manifest = json.loads((INPUT / "inputs.json").read_text(encoding="utf-8"))
    if file_hash(INPUT / "cold_features.parquet") != manifest["feature_hash"]:
        raise ValueError("cold feature hash changed")
    for source in manifest["cold_audit"]:
        if file_hash(DATA / source["file"]) != source["sha256"]:
            raise ValueError(f"training source changed: {source['file']}")
    frame = pd.read_parquet(INPUT / "cold_features.parquet")
    for column in CAT:
        frame[column] = frame[column].astype("category")
    frame.attrs["provenance"] = {"cold_feature_hash": manifest["feature_hash"],
                                  "raw_hashes": {source["file"]: source["sha256"] for source in manifest["cold_audit"]},
                                  "weather_hashes_at_resume": {str(path.relative_to(DATA)): file_hash(path)
                                                                for path in sorted((DATA / "weather").rglob("*.csv"))}}
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", default="1-7")
    parser.add_argument("--component", choices=("C", "all"), default="all")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    frame = inputs()
    months = tuple(int(value) for value in args.months.split("-"))
    fit, query = outer_masks(frame, months)
    fit &= frame.has_plan == 1
    query &= frame.has_plan == 1
    if args.component == "C":
        name = "ra_cb_wz_t7"
        fit_member(name, frame[fit], frame[query], OUT / args.months / "outer" / name)
    else:
        from champion_pipeline import run_block
        run_block(frame, months, OUT / args.months)


if __name__ == "__main__":
    main()
