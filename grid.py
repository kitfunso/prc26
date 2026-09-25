"""Every 1-, 2- and 3-member mix of the tuned members, NNLS and equal weights, each through the second stage and the
v34 gate. Files land in data/grid/<tag>.parquet, gate reads in data/grid/<tag>.gate.txt. `python grid.py [TAG ...]`."""
from __future__ import annotations

import itertools
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

from blend import load
from data import CACHE
from features import build
from train import HOLDOUT

M = {"C": "ra_cb_wz_t7", "L": "ra_lgb_pw_t7", "N": "nn3_pw_t7"}
OUT = Path("data/grid")


def weights(names: list[str], eq: bool) -> list[float]:
    if eq or len(names) == 1:
        return [1 / len(names)] * len(names)
    f = build()
    tr = f[(f["split"] == "train") & ~f["month"].isin(HOLDOUT)].merge(load(names, "oof"), on="id")
    w, _ = nnls(tr[names].values, tr["y"].values)
    return (w / w.sum()).tolist()


def plan() -> dict[str, tuple[list[str], bool]]:
    out = {}
    for k in (1, 2, 3):
        for c in itertools.combinations(M, k):
            out["".join(c)] = ([M[x] for x in c], False)
            if k > 1:
                out["".join(c) + "_eq"] = ([M[x] for x in c], True)
    return out


def run(tag: str, names: list[str], eq: bool) -> None:
    args = [f"{n}:{w:.6f}" for n, w in zip(names, weights(names, eq))]
    for step in ("cv", "fit"):
        subprocess.run([sys.executable, "recal.py", step, *args], check=True, capture_output=step == "fit")
    shutil.copy(CACHE / "recal_holdout.parquet", CACHE / f"recal_holdout_g{tag}.parquet")
    shutil.copy("data/recal_rank.parquet", OUT / f"{tag}.parquet")
    g = subprocess.run([sys.executable, "gate.py", f"recal_holdout_g{tag}.parquet", "recal_holdout_v34.parquet"], check=True, capture_output=True, text=True)
    (OUT / f"{tag}.gate.txt").write_text(" ".join(args) + "\n" + g.stdout, encoding="utf-8")
    print(f"{tag}: {' '.join(args)}\n{g.stdout}", flush=True)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    todo = plan()
    for tag in sys.argv[1:] or todo:
        run(tag, *todo[tag])
