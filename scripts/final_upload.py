"""Replace merry-octopus_final.parquet (the organisers allow changes until the deadline), read it back byte for byte,
then wait for the scorer's result. `python scripts/final_upload.py FILE`."""
from __future__ import annotations

import hashlib
import json
import sys
import time

from board import BUCKET, _s3

KEY = "merry-octopus_final.parquet"


def main(path: str) -> None:
    s3, data = _s3(), open(path, "rb").read()
    for key in ("final_ranking.parquet", "final_submitting.parquet"):
        print("dataset", key, s3.head_object(Bucket="prc-2026-datasets", Key=key)["LastModified"])
    old = s3.get_object(Bucket=BUCKET, Key=KEY)["Body"].read()
    print("replacing", KEY, len(old), "bytes, sha256", hashlib.sha256(old).hexdigest())
    s3.put_object(Bucket=BUCKET, Key=KEY, Body=data)
    if s3.get_object(Bucket=BUCKET, Key=KEY)["Body"].read() != data:
        raise RuntimeError("stored final differs from the local file")
    print("stored", KEY, len(data), "bytes, sha256", hashlib.sha256(data).hexdigest())
    started = time.time()
    while time.time() - started < 1800:
        time.sleep(30)
        try:
            obj = s3.get_object(Bucket=BUCKET, Key=KEY + "_result.json")
        except s3.exceptions.NoSuchKey:
            continue
        if obj["LastModified"].timestamp() > started - 5:
            print(json.dumps(json.loads(obj["Body"].read()), indent=1))
            return
    print("no new scorer result within 30 minutes")


if __name__ == "__main__":
    main(sys.argv[1])
