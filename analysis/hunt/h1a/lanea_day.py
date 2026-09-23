"""Bet case for lane A rows whose schedule delay tops 12 h (holdout and rank). 2025 evidence: every dev lane A
departure where the schedule or a flight-plan off-block (LOBT, IOBT, EOBT_1, AOBT_3) sits more than 12 h before
takeoff, with the label's class: copy of that stamp (|y - gap| <= 60 s), day-sized, or a normal taxi.
Run: PYTHONPATH=. python analysis/hunt/h1a/lanea_day.py"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import movements, v34_dev, v34_holdout, v34_rank

STAMPS = ["SCHED_TIME_UTC_mvt", "LOBT_flt", "IOBT_flt", "EOBT_1_flt", "AOBT_3_flt"]
BIG = 43200


def main() -> None:
    df = movements()
    d = df[(df["PHASE_mvt"] == "DEP") & df["has_plan"].astype(bool)].copy()
    for s in STAMPS:
        d["g_" + s[:4]] = d["MVT_TIME_UTC_mvt"] - d[s]
    gaps = ["g_" + s[:4] for s in STAMPS]
    d["part"] = np.where(d["dev"], "dev", np.where(d["ho"], "holdout", "rank"))
    pred = pd.concat([v34_dev()[["id", "p"]], v34_holdout()[["id", "p"]], v34_rank()])
    d = d.merge(pred, left_on="MVT_ID_mvt", right_on="id", how="left")
    far = d[(d[gaps] > BIG).any(axis=1)].copy()
    far["cls"] = np.select([far["y"].isna(), far["y"] > BIG, (far[gaps].sub(far["y"], axis=0)).abs().min(axis=1) <= 60],
                           ["?", "day-sized", "copy of a stamp"], "normal taxi")
    pd.set_option("display.width", 300)
    far["which"] = far[gaps].gt(BIG).apply(lambda r: "+".join(c[2:] for c in gaps if r[c]), axis=1)
    print(far.groupby(["part", "which", "cls"]).size().unstack(fill_value=0).to_string())
    dv = far[far["part"] == "dev"]
    print("\ndev lane A rows with a stamp > 12 h before takeoff, by airport:\n",
          dv.groupby(["apt", "cls"]).size().unstack(fill_value=0).to_string())
    print("\ndev y and v34 p on those rows:", dv["y"].describe().round(0).to_dict(), dv["p"].describe().round(0).to_dict())
    show = far[(far["apt"] == "LIRF") | ((far["part"] != "dev") & (far["p"] > 20000))]
    cols = ["part", "MVT_ID_mvt", "apt", "FLIGHT_mvt", "CALLSIGN_flt", "AIRCRAFT_OPERATOR_flt", "AIRCRAFT_TYPE_mvt",
            "STAND_mvt"] + gaps + ["y", "p"]
    print("\nLIRF rows, and holdout or rank rows where v34 predicts over 20000 s:\n", show[cols].round(0).to_string())


if __name__ == "__main__":
    main()
