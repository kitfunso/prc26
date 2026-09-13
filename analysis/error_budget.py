"""Where a blend's holdout squared error sits, by lane and stamp class (|y - z| < 60 s = copy), lane B rows from
laneb_holdout. `python analysis/error_budget.py [BLEND]`, default v26's lane A blend."""
import sys

import numpy as np
import pandas as pd

R = "C:/Users/skf_s/prc26/"
SKIP = {"id", "y", "split", "apt", "has_plan", "day", "month", "lane"}
BLEND = sys.argv[1] if len(sys.argv) > 1 else "blend_ra_cb_wz+ra_lgb_wz+nn3_wz"


def vec(name: str) -> pd.Series:
    d = pd.read_parquet(R + name)
    cols = [c for c in d.columns if c not in SKIP]
    assert "id" in d.columns and len(cols) == 1, (name, d.columns.tolist())
    return d.set_index("id")[cols[0]].astype(float)


def main() -> None:
    f = pd.read_parquet(R + "data/cache/feat.parquet", columns=["id", "has_plan", "y", "z_aobt", "z_eobt", "z_sched"])
    assert f["id"].is_unique
    f = f.set_index("id")
    p = vec(f"data/cache/{BLEND}_holdout.parquet")
    lb = vec("data/cache/laneb_holdout.parquet")
    ix = lb.index.intersection(p.index)
    p.loc[ix] = lb.loc[ix].values
    h = f.loc[p.index]
    y = h["y"]
    e2 = (p - y) ** 2
    n, S = len(e2), e2.sum()
    A = (h["has_plan"] == 1).values
    print(f"{BLEND} holdout rows {n}, rmse {np.sqrt(S / n):.2f}, lane A {np.sqrt(e2[A].mean()):.2f}, lane B {np.sqrt(e2[~A].mean()):.2f}")
    cls = np.select([(y - h.z_aobt).abs().lt(60), (y - h.z_eobt).abs().lt(60), (y - h.z_sched).abs().lt(60)],
                    ["AOBT copy", "EOBT copy", "sched copy"], "physical")
    c = (y - h.z_aobt).abs()
    kind = np.where(c.isna(), "physical, no AOBT", np.where(c > 600, "physical tail (>10 min off AOBT)", "physical bulk"))
    g = pd.DataFrame({"lane": np.where(A, "A", "B"), "cls": np.where(cls == "physical", kind, cls), "e2": e2.values})
    t = g.groupby(["lane", "cls"])["e2"].agg(["size", "sum"])
    t["rmse"] = np.sqrt(t["sum"] / t["size"]).round(0)
    t["rows%"] = (100 * t["size"] / n).round(2)
    t["sse%"] = (100 * t["sum"] / S).round(1)
    print(t.drop(columns="sum").to_string())
    s = np.sort(e2.values)[::-1]
    for k in (6, 25, 100, 1000):
        print(f"top {k} rows ({100 * k / n:.3f}% of rows) carry {100 * s[:k].sum() / S:.1f}% of squared error")
    big = ((p - y).abs() > 3600).values
    print(f"errors over 1 h: {big.sum()} rows, {100 * e2[big].sum() / S:.1f}% of squared error, {(~A[big]).sum()} of them lane B")


if __name__ == "__main__":
    main()
