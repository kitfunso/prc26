"""Airport-day disruption regime. r = the day's mean lane A prediction over the airport's 2025 norm; `hot` rows are
the r >= 1.35 slice every gate reports beside its headline (docs/EXPERIMENT-PROTOCOL.md). `python regime.py` writes
docs/REGIME.md and freezes the band map at data/cache/regime.parquet."""
from __future__ import annotations

import numpy as np
import pandas as pd

from data import CACHE, rmse
from train import boot_delta

BANDS = (0.0, 1.1, 1.2, 1.35, 1.5, 1.75, np.inf)
HOT = 1.35
RANK = "data/merry-octopus_v32.parquet"
DOC = "docs/REGIME.md"


def ratio(rank_file: str = RANK) -> pd.DataFrame:
    """Lane A rows with their airport-day r and band. The norm is the airport's median 2025 day-mean, so a month
    that is disrupted end to end cannot normalise itself away."""
    f = pd.read_parquet(CACHE / "feat.parquet", columns=["id", "split", "month", "day", "apt", "has_plan", "y"])
    f = f[(f["has_plan"] == 1) & f["split"].isin(("train", "rank"))].reset_index(drop=True)
    f["apt"] = f["apt"].astype(str)
    p = pd.concat([pd.read_parquet(CACHE / "recal_oof.parquet").rename(columns={"oof": "p"}),
                   pd.read_parquet(CACHE / "recal_holdout.parquet"),
                   pd.read_parquet(rank_file).rename(columns={"MVT_ID_mvt": "id", "TAXITIME_SEC_mvt": "p"})])
    f["p"] = f["id"].map(p.set_index("id")["p"])
    assert f["p"].notna().all(), "lane A row with no prediction in the regime source file"
    d = f.groupby(["split", "apt", "day"], observed=True)["p"].mean().rename("day_mean").reset_index()
    norm = d[d["split"] == "train"].groupby("apt", observed=True)["day_mean"].median()
    d["r"] = d["day_mean"] / d["apt"].map(norm).values
    out = f.merge(d[["split", "apt", "day", "r"]], on=["split", "apt", "day"], how="left")
    out["band"] = pd.cut(out["r"], BANDS)
    return out


def bands(refresh: bool = False) -> pd.DataFrame:
    """id -> r, frozen on first build: the slice must not move under the gate when the model changes."""
    out = CACHE / "regime.parquet"
    if out.exists() and not refresh:
        return pd.read_parquet(out)
    r = ratio()[["id", "r"]]
    r.to_parquet(out)
    return r


def hot(ids: np.ndarray) -> np.ndarray:
    return pd.Series(ids).map(bands().set_index("id")["r"]).values >= HOT


def slice_delta(y: np.ndarray, p0: np.ndarray, p1: np.ndarray, day: np.ndarray, ids: np.ndarray) -> str:
    """The gate line: paired day-block delta on the disrupted slice and on the rest, so a change cannot pass on
    calm rows alone while losing the 5% of rows that carry a quarter of the error."""
    h = hot(ids)
    parts = []
    for name, m in (("hot", h), ("calm", ~h)):
        if m.sum():
            d = boot_delta(y[m], p0[m], p1[m], day[m])
            parts.append(f"{name} n={int(m.sum())} {d[0]:+.2f} [{d[1]:+.2f}, {d[2]:+.2f}]")
    return " | ".join(parts)


def _table(g: pd.DataFrame, ref: pd.Series | None = None) -> pd.DataFrame:
    """Train side (ref None) measures RMSE; rank side projects SSE at the train band RMSE, which sizes the prize."""
    a = g.groupby("band", observed=True).agg(n=("id", "size"), r_max=("r", "max"))
    a["share_n%"] = (100 * a["n"] / a["n"].sum()).round(2)
    if ref is None:
        a["rmse"] = g.groupby("band", observed=True).apply(lambda d: rmse(d["y"].values, d["p"].values), include_groups=False)
        e = g.assign(se=(g["y"] - g["p"]) ** 2).groupby("band", observed=True)["se"].sum()
        a["sse%"] = (100 * e / e.sum()).round(2)
    else:
        a["proj_sse_bn"] = (a["n"] * ref.reindex(a.index).values ** 2 / 1e9).round(2)
        a["proj_sse%"] = (100 * a["proj_sse_bn"] / a["proj_sse_bn"].sum()).round(2)
    return a.round(2)


def main() -> None:
    g = ratio()
    tr, rk = g[g["split"] == "train"], g[g["split"] == "rank"]
    days = g.drop_duplicates(["split", "apt", "day"])
    t = _table(tr)
    lines = ["# Disruption regime (airport-day)\n",
             f"\nr = the airport-day's mean lane A prediction over the airport's median 2025 day-mean, from `{RANK}`",
             f" and the v32 out-of-fold predictions. Frozen at `data/cache/regime.parquet`; `hot` = r >= {HOT}.\n",
             "\n## Train, out-of-fold, by band\n\n", t.to_markdown(),
             "\n\n## Ranking rows by band\n\n", _table(rk, t["rmse"]).to_markdown(),
             "\n\n## Airport-days at r >= 1.5\n\n",
             days[days["r"] >= 1.5].groupby("split", observed=True).agg(days=("r", "size"), r_max=("r", "max")).round(2).to_markdown(),
             "\n\n## Ranking days above anything in training\n\n",
             rk[rk["r"] > days[days["split"] == "train"].groupby("apt", observed=True)["r"].max().reindex(rk["apt"]).values]
             .drop_duplicates(["apt", "day"])[["apt", "day", "r"]].sort_values("r", ascending=False).round(2).to_markdown(index=False), "\n"]
    open(DOC, "w", encoding="utf-8").writelines(lines)
    bands(refresh=True)
    print("".join(lines))


if __name__ == "__main__":
    main()
