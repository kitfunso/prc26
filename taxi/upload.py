"""Upload a submission file to the team bucket and check the stored bytes (protocol section 8)."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import boto3
from botocore.config import Config

ENDPOINT = "https://s3.opensky-network.org"
BUCKET = "prc-2026-merry-octopus"


def _keys() -> tuple[str, str]:
    env = dict(os.environ)
    if "PRC_KEY" not in env and Path(".env").exists():
        for line in Path(".env").read_text(encoding="utf-8").splitlines():
            k, _, v = line.partition("=")
            env.setdefault(k.strip(), v.strip())
    return env["PRC_KEY"], env["PRC_SECRET"]


def client():
    key, secret = _keys()
    return boto3.client("s3", endpoint_url=ENDPOINT, aws_access_key_id=key, aws_secret_access_key=secret,
                        region_name="us-east-1", config=Config(s3={"addressing_style": "path"}))


def listing(bucket: str = BUCKET) -> list[tuple[str, int]]:
    page = client().list_objects_v2(Bucket=bucket)
    return [(o["Key"], o["Size"]) for o in page.get("Contents", [])]


def upload(path: str, bucket: str = BUCKET) -> dict[str, object]:
    """Single put, then a head: the ETag must equal the local md5 and the size must match, or raise."""
    data = Path(path).read_bytes()
    md5, sha = hashlib.md5(data).hexdigest(), hashlib.sha256(data).hexdigest()
    s3, key = client(), Path(path).name
    s3.put_object(Bucket=bucket, Key=key, Body=data)
    head = s3.head_object(Bucket=bucket, Key=key)
    etag = head["ETag"].strip('"')
    if head["ContentLength"] != len(data) or etag != md5:
        raise RuntimeError(f"stored object differs: size {head['ContentLength']} vs {len(data)}, etag {etag} vs md5 {md5}")
    return {"bucket": bucket, "key": key, "bytes": len(data), "sha256": sha, "etag": etag}


def result(key: str, bucket: str = BUCKET) -> dict[str, object] | None:
    """The scorer writes <key>_result.json beside the file (score, used_pairs, or a rejection); None until it has run."""
    s3 = client()
    try:
        return json.loads(s3.get_object(Bucket=bucket, Key=f"{key}_result.json")["Body"].read())
    except s3.exceptions.NoSuchKey:
        return None


if __name__ == "__main__":
    if sys.argv[1:2] == ["list"]:
        print(listing())
    elif sys.argv[1:2] == ["status"]:
        for key, _ in listing():
            if key.endswith(".parquet"):
                print(key, result(key))
    else:
        print(upload(sys.argv[1]))
