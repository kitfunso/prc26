from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import laneb
from features import build
from train import HOLDOUT, boot_delta

OUT = ROOT / "data/runs/cp0-laneb-scope-20261003"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    sse = float(np.sum((y - p) ** 2))
    return {"sse": sse, "rmse": float(np.sqrt(sse / len(y)))}


def preserve_champion() -> dict:
    champion = ROOT / "data/merry-octopus_v53.parquet"
    template = pd.read_parquet(ROOT / "data/submitting.parquet")
    frame = pd.read_parquet(champion)
    if list(frame.columns) != list(template.columns) or not frame["MVT_ID_mvt"].is_unique:
        raise ValueError("champion schema or ID uniqueness failed")
    if set(frame["MVT_ID_mvt"]) != set(template["MVT_ID_mvt"]):
        raise ValueError("champion template ID set differs")
    p = frame["TAXITIME_SEC_mvt"].to_numpy(dtype=float)
    if not (np.isfinite(p).all() and (p >= 0).all()):
        raise ValueError("champion predictions failed")
    lane_b = pd.read_parquet(ROOT / "data/laneb_rank.parquet").set_index("MVT_ID_mvt")
    actual = frame.set_index("MVT_ID_mvt").loc[lane_b.index]
    pd.testing.assert_frame_equal(actual, lane_b)
    frozen = OUT / "champion_v53.parquet"
    if frozen.exists() and sha256(frozen) != sha256(champion):
        raise ValueError("existing frozen champion differs")
    if not frozen.exists():
        shutil.copy2(champion, frozen)
    members = {path.name: sha256(path) for path in (ROOT / "data").glob("*_t7_rank.parquet")}
    return {"path": str(champion), "sha256": sha256(champion), "rows": len(frame),
            "frozen_path": str(frozen), "frozen_sha256": sha256(frozen),
            "lane_b_rows": len(lane_b), "lane_b_exact_parity": True,
            "grid_cln_exact_bytes": sha256(champion) == sha256(ROOT / "data/grid/CLN.parquet"),
            "recorded_build_arguments": (ROOT / "data/grid/CLN.gate.txt").read_text().splitlines()[0],
            "member_prediction_hashes": members,
            "lineage_status": "prediction artifacts preserved; original model/checkpoint lineage incomplete"}


def main() -> None:
    start = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    champion = preserve_champion()
    print("champion schema, IDs, finite predictions and Lane B parity passed", flush=True)
    inputs = {path.name: sha256(path) for path in sorted((ROOT / "data").glob("training_*.parquet"))}
    for name in ("ranking.parquet", "submitting.parquet"):
        inputs[name] = sha256(ROOT / "data" / name)
    f = laneb.augment(build())
    train = f["split"] == "train"
    development = train & ~f["month"].isin(HOLDOUT)
    fit_ids = pd.Index(f.loc[development, "id"])
    print("building legacy and corrected real-data history", flush=True)
    legacy = laneb.history_rows(f, pd.Index(f.loc[train, "id"]))
    corrected = laneb.history_rows(f, fit_ids)
    changed = f.copy()
    changed.loc[train & ~development, "y"] = -99999.0
    pd.testing.assert_frame_equal(laneb.copy_hist(f, fit_ids), laneb.copy_hist(changed, fit_ids))
    del changed
    evaluation = (corrected["split"] == "train") & corrected["month"].isin(HOLDOUT)
    holdout = corrected[evaluation]
    history_changed = (legacy.loc[holdout.index, laneb.HIST] != holdout[laneb.HIST]).any(axis=1)
    donor = pd.read_parquet(laneb.CACHE / "mix_cb_wx_holdout.parquet").set_index("id")["p"]
    lane_a = pd.read_parquet(laneb.CACHE / "recal_holdout_gCLN.parquet").set_index("id")["p"]
    whole = f[train & f["month"].isin(HOLDOUT)].copy()
    whole_base = lane_a.reindex(whole["id"]).to_numpy()
    results, predictions, whole_predictions = {}, {}, {}
    for tag, scoped in (("legacy", legacy), ("corrected", corrected)):
        tr = scoped[(scoped["split"] == "train") & ~scoped["month"].isin(HOLDOUT)]
        te = scoped.loc[holdout.index]
        raw, _, probability, _, _ = laneb.predict(tr, te, iters=(117, 238))
        mixed = laneb.blend_donor(raw, probability[:, 0], donor.reindex(te["id"]).to_numpy())
        p = laneb.hull(te, laneb.hull_prior(tr, te, mixed))
        predictions[tag] = p
        whole_p = pd.Series(whole_base, index=whole["id"])
        whole_p.loc[te["id"].values] = p
        if not np.isfinite(whole_p).all():
            raise ValueError("diagnostic whole-file cache has missing predictions")
        whole_predictions[tag] = whole_p.to_numpy()
        results[tag] = {"lane_b": metrics(te["y"].to_numpy(), p),
                        "whole_file": metrics(whole["y"].to_numpy(), whole_p.to_numpy()),
                        "month": {str(m): metrics(te.loc[te["month"] == m, "y"].to_numpy(), p[te["month"].to_numpy() == m]) for m in HOLDOUT}}
        pd.DataFrame({"id": te["id"].values, "p": p}).to_parquet(OUT / f"{tag}_holdout.parquet")
        print(tag, results[tag], flush=True)
    print("checking real-data training-only stopping budgets", flush=True)
    _, _, outer_p, selected = laneb.outer_predict(f, development, pd.Index(holdout["id"]))
    pd.DataFrame({"id": holdout["id"].values, "p": outer_p}).to_parquet(OUT / "outer_raw_holdout.parquet")
    raw_checks = {"training_rows": int(train.sum()), "development_rows": int(development.sum()),
                  "holdout_lane_b_rows": len(holdout), "changed_history_rows": int(history_changed.sum()),
                  "has_plan_without_finite_aobt": int(((f["has_plan"] == 1) & ~np.isfinite(f["z_aobt"])).sum())}
    whole_y = whole["y"].to_numpy()
    gain = (whole_y - whole_predictions["legacy"]) ** 2 - (whole_y - whole_predictions["corrected"]) ** 2
    day_gain = pd.Series(gain, index=whole.index).groupby(whole["day"]).sum()
    result = {"run_id": OUT.name, "raw_hashes": inputs, "champion": champion,
              "checks": raw_checks, "hidden_label_mutation": "PASS", "fixed_iterations": [117, 238],
              "training_only_selected_iterations": list(selected), "results": results,
              "lane_b_bootstrap_gain": boot_delta(holdout["y"].to_numpy(), predictions["legacy"], predictions["corrected"], holdout["day"].to_numpy(), n=2000),
              "whole_file_bootstrap_gain": boot_delta(whole_y, whole_predictions["legacy"], whole_predictions["corrected"], whole["day"].to_numpy(), n=2000),
              "gain_concentration_sse": {str(n): float(np.sort(gain)[::-1][:n].sum()) for n in (1, 10, 50)},
              "leave_largest_gain_day_sse": float(gain.sum() - day_gain.max()),
              "largest_gain_day": str(day_gain.idxmax()), "whole_file_rows": len(whole),
              "seconds": time.monotonic() - start,
              "interpretation": "diagnostic correction on reused data; donor/Lane A caches are fixed historical inputs; complete nested baseline remains outstanding"}
    (OUT / "manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
