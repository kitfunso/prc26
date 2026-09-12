"""Leaderboard snapshot, submission file check, upload and score readback.

python board.py lb            -> data/board.json + per-team bests
python board.py check FILE    -> row set matches the template, no NaN, no negatives
python board.py upload FILE   -> put to the team bucket (name must be merry-octopus_vN.parquet), verify bytes
python board.py status        -> every file in the bucket with its scorer result
"""
from __future__ import annotations

import collections
import datetime
import hashlib
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

API = "https://datacomp.opensky-network.org/api/competitions/bb3693e1-26bc-4a9e-8619-4fe78b4eab0c/leaderboard"
ENDPOINT, BUCKET, US = "https://s3.opensky-network.org", "prc-2026-merry-octopus", "merry-octopus"
SNAP = Path("data/board.json")


def lb() -> None:
    rows, cursor = [], None
    while True:
        with urllib.request.urlopen(API + (f"?cursor={cursor}" if cursor else ""), timeout=60) as r:
            page = json.load(r)
        rows += page["items"]; cursor = page.get("nextCursor")
        if not cursor:
            break
    best, n = {}, collections.Counter()
    for r in rows:
        n[r["teamName"]] += 1
        if r["teamName"] not in best or r["score"] < best[r["teamName"]]["score"]:
            best[r["teamName"]] = r
    board = [{"rank": i, "team": t, "score": r["score"], "subs": n[t], "at": r["processedAt"][:16]}
             for i, (t, r) in enumerate(sorted(best.items(), key=lambda kv: kv[1]["score"]), 1)]
    snap = {"fetched_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M"), "submissions": len(rows),
            "teams": len(best), "leader": board[0], "us": next((b for b in board if b["team"] == US), None), "board": board}
    SNAP.write_text(json.dumps(snap, indent=1), encoding="utf-8")
    print(f"{snap['fetched_utc']}Z {len(rows)} submissions, {len(best)} teams. leader {board[0]['team']} {board[0]['score']:.4f}; us {snap['us']}")
    for b in board[:15]:
        print(f"{b['rank']:3d}. {b['team']:28s} {b['score']:9.4f}  subs={b['subs']:4d}")


def check(path: str) -> pd.DataFrame:
    sub, tpl = pd.read_parquet(path), pd.read_parquet("data/submitting.parquet")
    assert list(sub.columns) == list(tpl.columns), (list(sub.columns), list(tpl.columns))
    assert len(sub) == len(tpl) and set(sub["MVT_ID_mvt"]) == set(tpl["MVT_ID_mvt"]), "row set differs from the template"
    assert sub["MVT_ID_mvt"].is_unique, "duplicate ids"
    p = sub["TAXITIME_SEC_mvt"].to_numpy(dtype=float)
    assert np.isfinite(p).all() and (p >= 0).all(), "NaN or negative predictions"
    print(f"{path}: {len(sub)} rows ok, mean {p.mean():.1f}, p50 {np.median(p):.0f}, max {p.max():.0f}, sha256 {hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]}")
    return sub


def _s3():
    import boto3
    from botocore.config import Config
    env = dict(os.environ)
    for line in Path(".env").read_text(encoding="utf-8").splitlines():
        k, _, v = line.partition("=")
        env.setdefault(k.strip(), v.strip())
    return boto3.client("s3", endpoint_url=ENDPOINT, aws_access_key_id=env["PRC_KEY"], aws_secret_access_key=env["PRC_SECRET"],
                        region_name="us-east-1", config=Config(s3={"addressing_style": "path"}))


def upload(path: str) -> None:
    assert re.fullmatch(rf"{US}_v\d+\.parquet", Path(path).name), "name must be merry-octopus_vN.parquet"
    check(path)
    data, s3, key = Path(path).read_bytes(), _s3(), Path(path).name
    s3.put_object(Bucket=BUCKET, Key=key, Body=data)
    head = s3.head_object(Bucket=BUCKET, Key=key)
    assert head["ContentLength"] == len(data) and head["ETag"].strip('"') == hashlib.md5(data).hexdigest(), "stored bytes differ"
    print("uploaded", key, len(data), "bytes")


def status() -> None:
    s3 = _s3()
    for o in s3.list_objects_v2(Bucket=BUCKET).get("Contents", []):
        if o["Key"].endswith(".parquet"):
            try:
                res = json.loads(s3.get_object(Bucket=BUCKET, Key=o["Key"] + "_result.json")["Body"].read())
            except s3.exceptions.NoSuchKey:
                res = None
            print(o["Key"], res)


if __name__ == "__main__":
    {"lb": lb, "check": lambda: check(sys.argv[2]), "upload": lambda: upload(sys.argv[2]), "status": status}[sys.argv[1]]()
