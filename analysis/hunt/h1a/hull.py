"""Bet case for the LIRF lane B hull (zs >= 4 h). Classes on dev by zs group (4-12 h, > 12 h) and flight-number shape
(plain AAA123 vs garbled: doubled prefix, none, other). Strategy H: p = Pc * zs + (1 - Pc) * alt, Pc = Beta(1,1)
posterior copy rate of the row's cell on the other folds, alt = mean physical taxi + 86400 * round(delay / 1 day).
Rank risk: SSE change against v34 if every touched row is a copy, or every one is the alternative.
Run: PYTHONPATH=. python analysis/hunt/h1a/hull.py  (needs copyprob_lirf.parquet)"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import OUT
from copymodel import FOLDS

DAY = 86400
HULL = 14400


def cells(r: pd.DataFrame) -> pd.DataFrame:
    b = r[(r["lane"] == "B") & (r["zs"] >= HULL)].copy()
    b["shape"] = np.where(b["fmt"] == "plain", "plain", "garbled")
    b["zg"] = np.where(b["zs"] > 43200, ">12h", "4-12h")
    b["cls"] = np.select([b["y"].isna(), b["copy"] == 1, b["y"] > 43200], ["?", "copy", "shift"], "phys")
    return b


def alt(b: pd.DataFrame, taxi: float) -> pd.Series:
    return taxi + DAY * np.round((b["zs"] - taxi) / DAY)


def strategy(train: pd.DataFrame, apply: pd.DataFrame, taxi: float) -> pd.Series:
    g = train.groupby(["zg", "shape"])["copy"].agg(["sum", "count"])
    key = pd.MultiIndex.from_frame(apply[["zg", "shape"]])
    s = pd.Series(g["sum"].reindex(key).fillna(0).values, index=apply.index)
    n = pd.Series(g["count"].reindex(key).fillna(0).values, index=apply.index)
    pc = (s + 1) / (n + 2)
    return pc * apply["zs"] + (1 - pc) * alt(apply, taxi)


def main() -> None:
    r = pd.read_parquet(OUT / "copyprob_lirf.parquet")
    lb = r[(r["lane"] == "B") & (r["part"] == "dev")]
    taxi = float(lb[(lb["copy"] == 0) & (lb["y"] <= 3600)]["y"].mean())
    b = cells(r)
    pd.set_option("display.width", 250)
    print(f"mean physical taxi, LIRF lane B dev: {taxi:.0f} s")
    print(b.groupby(["part", "zg", "shape", "cls"]).size().unstack(fill_value=0).to_string())
    dv = b[b["part"] == "dev"].copy()
    for f in FOLDS:
        m = dv["month"].isin(f)
        dv.loc[m, "pH"] = strategy(dv[~m], dv[m], taxi)
    for name, s in (("all hull", dv), ("4-12h", dv[dv["zg"] == "4-12h"]), (">12h", dv[dv["zg"] == ">12h"])):
        e0, e1 = ((s["y"] - s["p"]) ** 2).sum() / 1e9, ((s["y"] - s["pH"]) ** 2).sum() / 1e9
        print(f"dev OOF {name}: n={len(s)} SSE v34 {e0:.3f}B strategy H {e1:.3f}B")
    for (zg, sh), s in dv.groupby(["zg", "shape"]):
        e0, e1 = ((s["y"] - s["p"]) ** 2).sum() / 1e9, ((s["y"] - s["pH"]) ** 2).sum() / 1e9
        print(f"  {zg} {sh}: n={len(s)} SSE v34 {e0:.3f}B H {e1:.3f}B")
    rk = b[b["part"] != "dev"].copy()
    rk["pH"] = strategy(dv, rk, taxi)
    rk["alt"] = alt(rk, taxi)
    for (part, zg, sh), s in rk.groupby(["part", "zg", "shape"]):
        d_copy = (((s["zs"] - s["p"]) ** 2).sum() - ((s["zs"] - s["pH"]) ** 2).sum()) / 1e9
        d_alt = (((s["alt"] - s["p"]) ** 2).sum() - ((s["alt"] - s["pH"]) ** 2).sum()) / 1e9
        print(f"{part} {zg} {sh}: n={len(s)} mean p/zs v34 {(s['p'] / s['zs']).mean():.3f} H {(s['pH'] / s['zs']).mean():.3f}"
              f" | SSE saved if all copies {d_copy:+.3f}B, if all alternative {d_alt:+.3f}B")
    rk.loc[rk["part"] == "rank", ["MVT_ID_mvt", "zs", "zg", "shape", "p", "pH", "alt"]].to_parquet(OUT / "hull_rank.parquet")


if __name__ == "__main__":
    main()
