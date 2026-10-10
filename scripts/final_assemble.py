"""Final-population file: Feb+Jun from this worktree's v53-recipe refit, Jan+Jul from the scored v53 file moved by each
row's takeoff re-jitter (final_ranking vs ranking), checked against final_submitting.parquet.
`python scripts/final_assemble.py V53 OLD_RANKING OUT [REFIT, default data/recal_rank.parquet]`."""
from __future__ import annotations

import hashlib
import sys

import numpy as np
import pandas as pd

ID, Y, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt", "MVT_TIME_UTC_mvt"


def rms(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(a))))


def main(v53_path: str, old_rank_path: str, out: str, refit: str = "data/recal_rank.parquet") -> None:
    tpl = pd.read_parquet("data/submitting.parquet")
    new = pd.read_parquet("data/ranking.parquet", columns=[ID, "PHASE_mvt", T])
    old = pd.read_parquet(old_rank_path, columns=[ID, "PHASE_mvt", T])
    new, old = new[new["PHASE_mvt"] == "DEP"].set_index(ID)[T], old[old["PHASE_mvt"] == "DEP"].set_index(ID)[T]
    v53 = pd.read_parquet(v53_path).set_index(ID)[Y]
    fresh = pd.read_parquet(refit).set_index(ID)[Y]
    if not (tpl[ID].is_unique and set(tpl[ID]) == set(new.index) == set(fresh.index)):
        raise ValueError("template, final ranking and refit IDs differ")
    if not set(v53.index) == set(old.index) <= set(new.index):
        raise ValueError("v53 does not cover exactly the original ranking departures")

    jj = v53.index
    delta = (new.loc[jj] - old.loc[jj]).dt.total_seconds()
    shifted = np.maximum(v53 + delta, 0)
    f = fresh.loc[jj]
    to_v53, to_shift = rms((f - v53).values), rms((f - shifted).values)
    use_shift = to_shift < to_v53
    print(f"Jan+Jul {len(jj)} rows: re-jitter sd {delta.std():.2f} s; refit vs v53 {to_v53:.2f} s, vs shifted v53 {to_shift:.2f} s"
          f" -> {'shifted' if use_shift else 'unshifted'} v53")

    pred = fresh.copy()
    pred.loc[jj] = (shifted if use_shift else v53).values
    sub = pd.DataFrame({ID: tpl[ID].values, Y: pred.loc[tpl[ID].values].values.astype(tpl[Y].dtype)})
    if not (np.isfinite(sub[Y]).all() and (sub[Y] >= 0).all() and len(sub) == len(tpl) and list(sub.columns) == list(tpl.columns)):
        raise ValueError("final file fails the template check")
    sub.to_parquet(out, index=False)

    month = pd.Series(new.loc[sub[ID]].dt.month.values, index=sub[ID].values)
    print(sub.groupby(month.values)[Y].agg(["size", "mean", "median"]).round(1).to_string())
    print(out, len(sub), hashlib.sha256(open(out, "rb").read()).hexdigest())


if __name__ == "__main__":
    main(*sys.argv[1:5])
