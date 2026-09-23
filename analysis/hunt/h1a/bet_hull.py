"""The one hull bet: LIRF lane B departures with 4 h <= zs <= 12 h whose flight number is garbled (doubled prefix such
as ITYTY672 / ARGG1141, missing, or other non-AAA123 shapes) are schedule copies, so y = zs. Evidence and the
prediction come from the ten dev months only: q = Beta(1,1) posterior copy rate of the dev rows in the cell,
p = q * zs + (1 - q) * mean physical LIRF lane B taxi. Writes bet overrides for rank and holdout (touched rows only).
Run: PYTHONPATH=. python analysis/hunt/h1a/bet_hull.py"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import OUT
from copymodel import lirf_frame

ZLO, ZHI = 14400, 43200


def touched(d: pd.DataFrame) -> pd.Series:
    return ((d["PHASE_mvt"] == "DEP") & (d["lane"] == "B") & d["zs"].between(ZLO, ZHI) & (d["fmt"] != "plain"))


def main() -> None:
    d = lirf_frame()
    lb = d[(d["PHASE_mvt"] == "DEP") & (d["lane"] == "B") & (d["part"] == "dev")]
    taxi = float(lb[(lb["copy"] == 0) & (lb["y"] <= 3600)]["y"].mean())
    t = d[touched(d)].copy()
    dev = t[t["part"] == "dev"]
    k, n = int(dev["copy"].sum()), len(dev)
    q = (k + 1) / (n + 2)
    t["p_bet"] = q * t["zs"] + (1 - q) * taxi
    print(f"dev cell: {k} copies of {n} rows; q = {q:.4f}; mean physical taxi {taxi:.0f} s")
    print(f"touched rows: {t.groupby('part').size().to_dict()}")
    dev = t[t["part"] == "dev"]
    print(f"dev SSE on the cell (in-sample q): v34 {((dev['y'] - dev['p']) ** 2).sum() / 1e9:.3f}B,"
          f" bet {((dev['y'] - dev['p_bet']) ** 2).sum() / 1e9:.3f}B")
    for part in ("holdout", "rank"):
        s = t[t["part"] == part]
        alt = taxi + 86400 * np.round((s["zs"] - taxi) / 86400)
        g_copy = (((s["zs"] - s["p"]) ** 2) - ((s["zs"] - s["p_bet"]) ** 2)).sum() / 1e9
        g_alt = (((alt - s["p"]) ** 2) - ((alt - s["p_bet"]) ** 2)).sum() / 1e9
        per_alt = (((alt - s["p"]) ** 2) - ((alt - s["p_bet"]) ** 2)).mean() / 1e9
        print(f"{part}: {len(s)} rows, SSE saved if all copies {g_copy:+.3f}B, if all physical {g_alt:+.3f}B"
              f" (mean cost per physical row {per_alt:+.3f}B)")
    pd.set_option("display.width", 250)
    s = t[t["part"] != "dev"].assign(sched=lambda x: pd.to_datetime(x["SCHED_TIME_UTC_mvt"], unit="s"))
    print(s[["part", "MVT_ID_mvt", "FLIGHT_mvt", "fmt", "STAND_mvt", "ADES_mvt", "sched", "zs", "p", "p_bet"]]
          .sort_values(["part", "sched"]).round(0).to_string())
    r = t[t["part"] == "rank"]
    r[["MVT_ID_mvt"]].assign(TAXITIME_SEC_mvt=r["p_bet"].values).to_parquet(OUT / "bet_hull_rank_override.parquet")
    h = t[t["part"] == "holdout"]
    h[["MVT_ID_mvt"]].rename(columns={"MVT_ID_mvt": "id"}).assign(p=h["p_bet"].values).to_parquet(
        OUT / "bet_hull_holdout_override.parquet")


if __name__ == "__main__":
    main()
