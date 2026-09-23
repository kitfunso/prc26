"""Every public board upload: rows scored per upload, and how fast each top-15 team reached its level.
Run from prc26: PYTHONPATH=. python analysis/board_history.py (writes data/board_rows.json)."""
import json
import urllib.request
from pathlib import Path

import pandas as pd

from board import API, US

rows, cursor = [], None
while True:
    with urllib.request.urlopen(API + (f"?cursor={cursor}" if cursor else ""), timeout=60) as r:
        page = json.load(r)
    rows += page["items"]; cursor = page.get("nextCursor")
    if not cursor:
        break
Path("data/board_rows.json").write_text(json.dumps(rows), encoding="utf-8")
d = pd.DataFrame(rows)
d["t"] = pd.to_datetime(d["processedAt"])
d = d.sort_values("t")
n = int(d["usedPairs"].iloc[0])
print(f"{len(d)} uploads, rows scored per upload: {d['usedPairs'].value_counts().to_dict()}, first {d['t'].min():%Y-%m-%d}")
for day in ("2026-09-14", "2026-09-17", "2026-09-20", None):
    s = (d if day is None else d[d["t"] < pd.Timestamp(day, tz="UTC") + pd.Timedelta(days=1)]).groupby("teamName")["score"].min().sort_values()
    print(f"end of {day or 'now'}: {len(s)} teams, 1st {s.iloc[0]:.2f}, 3rd {s.iloc[2]:.2f}, 10th {s.iloc[9]:.2f},"
          f" us {s[US]:.2f} at rank {s.index.get_loc(US) + 1}")
best = d.groupby("teamName")["score"].min().sort_values()
for team in list(best.index[:15]) + [US]:
    g = d[d["teamName"] == team].reset_index(drop=True)
    hit = g.index[g["score"] < 250]
    when = f"under 250 at upload {hit[0] + 1}, {(g['t'][hit[0]] - g['t'][0]).days} days in" if len(hit) else "never under 250"
    prev = g["score"].cummin().shift(1)
    drop = (prev - g["score"]).where(prev < 275)
    jump = "none"
    if drop.max() > 0:
        i = drop.idxmax()
        jump = f"{drop[i]:.2f} s from {prev[i]:.2f} on {g['t'][i]:%m-%d}, {n * (prev[i] ** 2 - g['score'][i] ** 2) / 1e9:.2f}B SSE"
    print(f"{best.index.get_loc(team) + 1:3d} {team:22s} best {g['score'].min():7.2f}, {len(g):3d} uploads, "
          f"first {g['t'][0]:%m-%d} at {g['score'][0]:7.2f}, {when}; largest gain from a best under 275: {jump}")
