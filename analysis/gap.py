"""Where the file of record's squared error sits on the Jan+Jul 2025 holdout, and what closing the board gap would take.
Run from prc26: PYTHONPATH=. python analysis/gap.py [OUR_BOARD_SCORE]. Needs data/board.json from `python board.py lb`."""
import json
import sys

import numpy as np
import pandas as pd

from data import CACHE
from laneb import APT_SHIFT, HULL_Z
from train import HOLDOUT

N_RANK, N_A, N_B = 344_841, 339_551, 5_290
HULL_SSE = 4.01e9  # expected SSE on the 43 LIRF hull rank rows, docs/LANEB.md section 12
us = float(sys.argv[1]) if len(sys.argv) > 1 else 270.5655

f = pd.read_parquet(CACHE / "feat.parquet", columns=["id", "split", "month", "has_plan", "y", "apt", "z_sched"])
h = f[(f["split"] == "train") & f["month"].isin(HOLDOUT)].reset_index(drop=True)
la = pd.read_parquet(CACHE / "recal_holdout_v34.parquet").set_index("id")["p"]
lb = pd.read_parquet(CACHE / "laneb_holdout.parquet").set_index("id")["p"]
a = h["has_plan"].values == 1
y = h["y"].values.astype(float)
e2 = (y - np.where(a, h["id"].map(la).values, h["id"].map(lb).values).astype(float)) ** 2
S = e2.sum()
r = lambda m: np.sqrt(e2[m].mean())
print(f"holdout rows {len(h)} (A {a.sum()}, B {(~a).sum()}); RMSE all {np.sqrt(e2.mean()):.2f}, A {r(a):.2f}, B {r(~a):.2f}")
print(f"SSE share: lane A {e2[a].sum() / S:.3f}, lane B {e2[~a].sum() / S:.3f} (lane B is {(~a).mean():.4f} of rows)")
cum = np.cumsum(np.sort(e2)[::-1]) / S
for k in (10, 43, 345, 3450):
    print(f"top {k} rows hold {cum[k - 1]:.3f} of SSE")
big = y > 43_200
print(f"labels over 12 h: {big.sum()} rows hold {e2[big].sum() / S:.3f} of SSE; lane B without them: "
      f"{(~a & ~big).sum()} rows, RMSE {r(~a & ~big):.1f}, SSE {e2[~a & ~big].sum() / 1e9:.3f}B")
by = pd.DataFrame({"lane_B": ~a, "apt": h["apt"].astype(str).values, "e2": e2}).groupby(["lane_B", "apt"])["e2"]
print((by.agg(["size", "sum"]).assign(share=lambda d: d["sum"] / S).sort_values("sum", ascending=False).head(6)).round(4))

board = {b["rank"]: b["score"] for b in json.load(open("data/board.json", encoding="utf-8"))["board"]}
sse = N_RANK * us ** 2
floor = N_RANK * (us ** 2 - (us - 1.5) ** 2)
print(f"\nboard SSE at {us}: {sse / 1e9:.3f}B; 1.5 s of board = {floor / 1e6:.1f}M SSE")
m = ~a & ~big & ~((h["apt"].astype(str).values == APT_SHIFT) & (h["z_sched"].values >= HULL_Z))
print(f"lane B outside the {APT_SHIFT} hull, labels at or under 12 h: {m.sum()} rows, RMSE {r(m):.1f},"
      f" SSE {e2[m].sum() / 1e9:.3f}B; the 1.5 s bar is RMSE {np.sqrt((e2[m].sum() - floor) / m.sum()):.1f}")
sa = N_A * r(a) ** 2  # assumption: board lane A runs at its holdout RMSE
sb = sse - sa
print(f"assumed board lane A SSE {sa / 1e9:.2f}B, implied lane B {sb / 1e9:.2f}B = RMSE {np.sqrt(sb / N_B):.0f} s")
for k in (1, 3, 9, 10, 20):
    g = sse - N_RANK * board[k] ** 2
    print(f"rank {k} {board[k]:.4f}: gap {g / 1e9:.3f}B; lane A alone {r(a):.1f} -> {np.sqrt((sa - g) / N_A):.1f};"
          f" lane B alone {np.sqrt(sb / N_B):.0f} -> {np.sqrt(max(sb - g, 0) / N_B):.0f}")
g = sse - HULL_SSE - N_RANK * board[3] ** 2
print(f"all 43 hull rows known: board {np.sqrt((sse - HULL_SSE) / N_RANK):.2f}; rank 3 then still needs {g / 1e9:.3f}B,"
      f" lane A {r(a):.1f} -> {np.sqrt((sa - g) / N_A):.1f}")
