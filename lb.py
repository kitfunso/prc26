"""Pull the full paginated leaderboard and print per-team best RMSE."""
import json, urllib.request, collections

API = ("https://datacomp.opensky-network.org/api/competitions/"
       "bb3693e1-26bc-4a9e-8619-4fe78b4eab0c/leaderboard")

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
    print(f"{len(rows)} submissions, {len(best)} teams with a score\n")
    for i, (t, r) in enumerate(sorted(best.items(), key=lambda kv: kv[1]["score"]), 1):
        print(f"{i:3d}. {t:28s} {r['score']:9.4f}  subs={n[t]:4d}  {r['processedAt'][:16]}")
