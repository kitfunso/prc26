"""FF: forward fold (train Jan-Nov 2025, predict Dec 2025) for every lane champion at its recorded rounds; reported, never decides."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from taxi.baselines import b4
from taxi.champ import CHAMPIONS_A, predict_b, refit_predict
from taxi.data import load
from taxi.m1 import MODEL_FEATURES, chosen_cells, final_rounds
from taxi.m2 import predict_m2, stages
from taxi.mc1 import predict_cb, stages_cb
from taxi.verdict import rmse
from taxi.w1 import COLS_A, m2_rounds
from taxi.weather import join, load_metar


def lane_a_pred(name: str, tr: pd.DataFrame, te: pd.DataFrame) -> np.ndarray:
    if name in CHAMPIONS_A:
        return refit_predict(name, tr, te)
    if name == "M2":
        return predict_m2(stages(tr, chosen_cells()["A"][0], rounds=m2_rounds(), cols=MODEL_FEATURES), te)[0]
    return predict_cb(stages_cb(tr, iters=final_rounds("MC1", "docs/MC1.md"), cols=COLS_A), te, COLS_A)[0]


def kept(lane: str) -> list[str]:
    """The other lane's rows from the file on disk, so one lane can be refreshed without refitting the other on the GPU."""
    try:
        rows = open("docs/FORWARD.md", encoding="utf-8").read().split("\n")
    except FileNotFoundError:
        return []
    return [r for r in rows if r.startswith(f"| {lane} |")]


def main(only: str | None = None) -> None:
    metar = load_metar()
    fwd, dec = (join(load(m), metar) for m in (tuple(range(1, 12)), (12,)))
    lines = ["| lane | model | Dec rows | RMSE (s) | fit s |", "|---|---|---|---|---|"]
    # The lane-B chain is named, not read from CURRENT: the file must show every promoted step, not only the one current when it last ran.
    for lane, names in (("A", ("M2", "W1-A", "D1", "MC1", "MC2")), ("B", ("B4", "L4b", "L6a", "L8c"))):
        if only and lane != only:
            lines += kept(lane)
            continue
        f, d = (x[x.has_fpl == (lane == "A")].reset_index(drop=True) for x in (fwd, dec))
        for name in names:
            t0 = time.time()
            p = lane_a_pred(name, f, d) if lane == "A" else (b4(f, d, "B") if name == "B4" else predict_b(name, f, d))
            lines.append(f"| {lane} | {name} | {len(d)} | {rmse(p - d.y.values):.1f} | {time.time() - t0:.0f} |")
            print(lines[-1], flush=True)
    text = "\n".join(["# Forward fold (reported, never decides): train Jan-Nov 2025, predict Dec 2025", "",
                      "Regenerate with `python -m taxi.fwd`. Every model at the rounds its report recorded; the M1 numbers are in `docs/M1.md`.", "",
                      *lines]) + "\n"
    with open("docs/FORWARD.md", "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else None)
