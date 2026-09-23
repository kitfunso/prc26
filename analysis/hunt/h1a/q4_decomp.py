"""Q4: how much of v34's development error on the big rows is the wrong CLASS and how much the wrong SIZE inside the
right class? Class from the label: copy (y = zs), kcopy (y = zs + k days), shift (physical + k days), phys.
Oracle size given the class: copy zs, kcopy zs + k days, shift k days + mean physical taxi, phys mean physical taxi
(mean over that airport and lane's physical rows with y <= 1 h). Class part = SSE(v34) - SSE(oracle), floored at 0.
Run: PYTHONPATH=. python analysis/hunt/h1a/q4_decomp.py"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common import movements, v34_dev
from origin import DAY


def label_class(y: pd.Series, zs: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Class and day count k of every labelled row."""
    kc = np.round((y - zs) / DAY)
    ks = np.round((y - 1200) / DAY)
    cls = pd.Series("phys", index=y.index)
    cls[(ks >= 1) & (y >= 43200)] = "shift"
    cls[(kc != 0) & ((y - zs - kc * DAY).abs() <= 10)] = "kcopy"
    cls[(y - zs).abs() <= 10] = "copy"
    k = pd.Series(0.0, index=y.index).where(~cls.isin(["kcopy", "shift"]), np.where(cls == "kcopy", kc, ks))
    return cls, k


def frame() -> pd.DataFrame:
    df = movements()
    d = df[(df["PHASE_mvt"] == "DEP") & df["dev"]].merge(v34_dev(), left_on="MVT_ID_mvt", right_on="id")
    d["zs"] = d["MVT_TIME_UTC_mvt"] - d["SCHED_TIME_UTC_mvt"]
    d["cls"], d["k"] = label_class(d["y"], d["zs"])
    tax = d[(d["cls"] == "phys") & (d["y"] <= 3600)].groupby(["apt", "lane"])["y"].mean().rename("taxi")
    d = d.join(tax, on=["apt", "lane"])
    d["q"] = np.select([d["cls"] == "copy", d["cls"] == "kcopy", d["cls"] == "shift"],
                       [d["zs"], d["zs"] + d["k"] * DAY, d["k"] * DAY + d["taxi"]], d["taxi"])
    d["e2"] = (d["y"] - d["p"]) ** 2
    d["o2"] = (d["y"] - d["q"]) ** 2
    d["yb"] = pd.cut(d["y"], [-1e9, 3600, 43200, 1e9], labels=["<=1h", "1-12h", ">12h"])
    return d


def main() -> None:
    d = frame()
    pd.set_option("display.width", 250)
    print(f"all dev rows: n={len(d)} SSE_v34={d['e2'].sum() / 1e9:.3f}B")
    big = d[d["y"] > 3600]
    g = big.groupby(["yb", "lane", "apt", "cls"], observed=True).agg(n=("y", "size"), sse_v34=("e2", "sum"),
                                                                     sse_oracle=("o2", "sum"))
    g["class_part"] = (g["sse_v34"] - g["sse_oracle"]).clip(lower=0)
    g["size_part"] = g["sse_v34"] - g["class_part"]
    g = g[g["sse_v34"] > 2e7]
    print((g[["sse_v34", "sse_oracle", "class_part", "size_part"]] / 1e9).round(3).assign(n=g["n"]).to_string())
    for yb in ("1-12h", ">12h"):
        s = g.xs(yb, level="yb")
        print(f"\ny {yb}: SSE_v34 {s['sse_v34'].sum() / 1e9:.3f}B, class part {s['class_part'].sum() / 1e9:.3f}B,"
              f" size part {s['size_part'].sum() / 1e9:.3f}B (groups with SSE > 0.02B)")
    lb = d[(d["apt"] == "LIRF") & (d["lane"] == "B")].copy()
    lb["zsb"] = pd.cut(lb["zs"] / 3600, [-99, 0.5, 1, 2, 4, 12, 999])
    lb["pimp"] = ((lb["p"] - lb["taxi"]) / (lb["zs"] - lb["taxi"])).where(lb["zs"] > 3600).clip(0, 1)
    t = lb.groupby("zsb", observed=True).agg(n=("y", "size"), copy_rate=("cls", lambda c: (c == "copy").mean()),
                                              v34_implied_pcopy=("pimp", "mean"), sse_v34=("e2", "sum"),
                                              sse_oracle=("o2", "sum"))
    t[["sse_v34", "sse_oracle"]] /= 1e9
    print("\nLIRF lane B by zs bin: copy rate vs v34's implied copy probability\n", t.round(3).to_string())
    la = d[(d["apt"] == "LIRF") & (d["lane"] == "A")].copy()
    la["zsb"] = pd.cut(la["zs"] / 3600, [-99, 0.5, 1, 2, 4, 12, 999])
    la["pimp"] = ((la["p"] - la["taxi"]) / (la["zs"] - la["taxi"])).where(la["zs"] > 3600).clip(0, 1)
    t = la.groupby("zsb", observed=True).agg(n=("y", "size"), copy_rate=("cls", lambda c: (c == "copy").mean()),
                                              v34_implied_pcopy=("pimp", "mean"), sse_v34=("e2", "sum"),
                                              sse_oracle=("o2", "sum"))
    t[["sse_v34", "sse_oracle"]] /= 1e9
    print("\nLIRF lane A by zs bin\n", t.round(3).to_string())


if __name__ == "__main__":
    main()
