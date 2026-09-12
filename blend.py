"""Non-negative least-squares blend of named configs on OOF, scored on holdout. `python blend.py cb0 lgb0 ...`"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from data import CACHE, rmse
from features import build
from train import HOLDOUT, RESULTS, boot_delta, report


def load(names: list[str], kind: str) -> pd.DataFrame:
    out = None
    for n in names:
        d = pd.read_parquet(CACHE / f"{n}_{kind}.parquet").rename(columns={"oof": n, "p": n})
        out = d if out is None else out.merge(d, on="id")
    return out


def main(names: list[str], base: str | None) -> None:
    f = build()
    f = f[f["split"] == "train"]
    tr = f[~f["month"].isin(HOLDOUT)].merge(load(names, "oof"), on="id")
    ho = f[f["month"].isin(HOLDOUT)].merge(load(names, "holdout"), on="id")
    w, _ = nnls(tr[names].values, tr["y"].values)
    w = w / w.sum()
    p_tr, p_ho = np.maximum(tr[names].values @ w, 0), np.maximum(ho[names].values @ w, 0)
    cv_all, ho_all = report(tr["y"].values, p_tr, tr), report(ho["y"].values, p_ho, ho)
    name = "blend:" + "+".join(names)
    pd.DataFrame({"id": ho["id"].values, "p": p_ho}).to_parquet(CACHE / f"{name.replace(':', '_')}_holdout.parquet")
    line = f"| {name} w={np.round(w, 3).tolist()} | {json.dumps(cv_all)} | {json.dumps(ho_all)} | - | - |\n"
    if base:
        b = ho.merge(pd.read_parquet(CACHE / f"{base}_holdout.parquet").rename(columns={"p": "base"}), on="id")
        d = boot_delta(b["y"].values, b["base"].values, p_ho, b["day"].values)
        line = line.rstrip("|\n") + f" vs {base}: {d[0]:+.2f} [{d[1]:+.2f}, {d[2]:+.2f}] |\n"
    RESULTS.open("a", encoding="utf-8").write(line)
    print(line)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--base=")]
    base = next((a[7:] for a in sys.argv[1:] if a.startswith("--base=")), None)
    main(args, base)
