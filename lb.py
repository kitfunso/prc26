"""Pull the full paginated leaderboard, print per-team best RMSE, and snapshot it to data/board.json.

The snapshot is the dated source every report reads: the board moves (the leader gained 11.5 s in two days),
so a number pasted into prose rots silently.
"""
import json, urllib.request, collections, datetime, pathlib

API = ("https://datacomp.opensky-network.org/api/competitions/"
       "bb3693e1-26bc-4a9e-8619-4fe78b4eab0c/leaderboard")
SNAP = pathlib.Path("data/board.json")
US = "merry-octopus"

def fetch_all() -> list[dict]:
    rows, cursor = [], None
    while True:
        url = API + (f"?cursor={cursor}" if cursor else "")
        with urllib.request.urlopen(url, timeout=60) as r:
            page = json.load(r)
        rows += page["items"]
        cursor = page.get("nextCursor")
        if not cursor:
            return rows

if __name__ == "__main__":
    rows = fetch_all()
    best, n = {}, collections.Counter()
    for r in rows:
        t = r["teamName"]
        n[t] += 1
        if t not in best or r["score"] < best[t]["score"]:
            best[t] = r
    order = sorted(best.items(), key=lambda kv: kv[1]["score"])
    board = [{"rank": i, "team": t, "score": r["score"], "subs": n[t], "at": r["processedAt"][:16]}
             for i, (t, r) in enumerate(order, 1)]
    snap = {
        "fetched_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "submissions": len(rows),
        "teams": len(best),
        "leader": board[0],
        "us": next((b for b in board if b["team"] == US), None),
        "board": board,
    }
    SNAP.parent.mkdir(exist_ok=True)
    SNAP.write_text(json.dumps(snap, indent=1), encoding="utf-8")
    print(f"{len(rows)} submissions, {len(best)} teams with a score -> {SNAP}\n")
    for b in board:
        print(f"{b['rank']:3d}. {b['team']:28s} {b['score']:9.4f}  subs={b['subs']:4d}  {b['at']}")
