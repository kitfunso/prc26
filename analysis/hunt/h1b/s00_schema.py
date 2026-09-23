"""H1b item 7 (part 1): dump parquet schema/metadata for the raw files and feat.parquet.
Metadata only -- no full column load. Run: PYTHONPATH=. python analysis/hunt/h1b/s00_schema.py
"""
from __future__ import annotations

import pyarrow.parquet as pq

FILES = [
    "data/training_2025-01-01_2025-02-01.parquet",
    "data/ranking.parquet",
    "data/submitting.parquet",
    "data/cache/feat.parquet",
]


def dump(path: str) -> None:
    pf = pq.ParquetFile(path)
    meta = pf.metadata
    print(f"\n=== {path} ===")
    print(f"rows={meta.num_rows} row_groups={meta.num_row_groups} created_by={meta.created_by!r}")
    kv = meta.metadata or {}
    print(f"kv_metadata_keys={list(kv.keys())}")
    for k, v in kv.items():
        if k not in (b"ARROW:schema",):
            print(f"  {k!r}: {v[:200]!r}")
    sch = pf.schema_arrow
    print(f"n_columns={len(sch)}")
    for f in sch:
        print(f"  {f.name}: {f.type}")
    # row group sizes and any per-column stats worth noting for the first row group
    rg0 = meta.row_group(0)
    print(f"row_group_0: num_rows={rg0.num_rows} total_byte_size={rg0.total_byte_size}")


if __name__ == "__main__":
    for p in FILES:
        dump(p)
