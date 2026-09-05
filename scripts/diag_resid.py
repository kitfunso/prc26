"""RD diagnostic (protocol: not evidence): where MC2's cached holdout squared error sits, by airport, runway, hour and airport x runway x hour."""
from __future__ import annotations

import numpy as np
import pandas as pd

from taxi.champ import CURRENT, champion_holdout, lane_a

OUT, TOP = "docs/diag_resid.md", 15


def table(df: pd.DataFrame, keys: list[str], e: np.ndarray, top: int = TOP) -> list[str]:
    g = df[keys].assign(se=e * e).groupby(keys, observed=True).se.agg(["count", "sum", "mean"])
    g["share"] = g["sum"] / g["sum"].sum()
    g["rmse"] = np.sqrt(g["mean"])
    g = g.sort_values("share", ascending=False).head(top).reset_index()
    head = f"| {' | '.join(keys)} | rows | SSE share | RMSE (s) |"
    rows = [f"| {' | '.join(str(r[k]) for k in keys)} | {int(r['count'])} | {r['share']:.3f} | {r['rmse']:.1f} |" for _, r in g.iterrows()]
    return [head, "|" + "---|" * (len(keys) + 3), *rows]


def main() -> None:
    tra, tea = lane_a()
    name = CURRENT["A"]
    e = champion_holdout(name, tra, tea) - tea.y.values
    out = [f"# RD: where {name}'s holdout squared error sits (diagnostic, not evidence)", "",
           f"Regenerate with `python scripts/diag_resid.py`. {len(tea)} lane-A holdout rows, RMSE {np.sqrt((e * e).mean()):.1f} s, "
           f"mean error {e.mean():+.1f} s. Share = the group's sum of squared errors over the total; top {TOP} groups per table.", ""]
    for keys in (["apt"], ["apt", "runway"], ["hour"], ["apt", "runway", "hour"]):
        out += [f"## by {' x '.join(keys)}", "", *table(tea, keys, e), ""]
    text = "\n".join(out)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
