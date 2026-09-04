"""Day-1 probe: list the challenge buckets, dump the schema, test the obvious leak.

Needs MinIO keys in env: PRC_KEY / PRC_SECRET (generate at opensky-network.org, then
get your OSN account added to the buckets by registering a team).
"""
import os
import duckdb

BUCKET = os.environ.get("PRC_BUCKET", "competition-data")


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute(f"""
        CREATE SECRET osn (
            TYPE s3,
            KEY_ID '{os.environ["PRC_KEY"]}',
            SECRET '{os.environ["PRC_SECRET"]}',
            ENDPOINT 's3.opensky-network.org',
            URL_STYLE 'path',
            USE_SSL true
        )
    """)
    return con


def main() -> None:
    con = connect()
    rank = f"s3://{BUCKET}/ranking.parquet"

    print("== ranking schema (which columns survived the blanking) ==")
    print(con.sql(f"DESCRIBE SELECT * FROM '{rank}'").df().to_string())

    print("\n== non-null counts for departures ==")
    cols = [r[0] for r in con.sql(f"DESCRIBE SELECT * FROM '{rank}'").fetchall()]
    counts = ", ".join(f"count({c}) AS {c}" for c in cols)
    print(con.sql(
        f"SELECT count(*) AS n, {counts} FROM '{rank}' WHERE PHASE_mvt='DEP'"
    ).df().T.to_string())

    # The whole challenge is takeoff - offblock. Takeoff is given. If NM's actual
    # off-block (AOBT_3_flt) survived the blanking, the target is arithmetic.
    print("\n== leak check: MVT_TIME - AOBT_3_flt vs true taxi-out (on training) ==")
    train = f"s3://{BUCKET}/training_2025-07-01_2025-08-01.parquet"
    print(con.sql(f"""
        SELECT count(*) AS n,
               avg(epoch(MVT_TIME_UTC_mvt) - epoch(AOBT_3_flt) - TAXITIME_SEC_mvt) AS bias_s,
               stddev(epoch(MVT_TIME_UTC_mvt) - epoch(AOBT_3_flt) - TAXITIME_SEC_mvt) AS rmse_ish_s,
               median(abs(epoch(MVT_TIME_UTC_mvt) - epoch(AOBT_3_flt) - TAXITIME_SEC_mvt)) AS mae_med_s
        FROM '{train}'
        WHERE PHASE_mvt='DEP' AND AOBT_3_flt IS NOT NULL AND TAXITIME_SEC_mvt IS NOT NULL
    """).df().to_string())

    print("\n== target distribution (the RMSE we must beat is ~277s) ==")
    print(con.sql(f"""
        SELECT ADEP_mvt, count(*) AS n,
               avg(TAXITIME_SEC_mvt) AS mean_s, stddev(TAXITIME_SEC_mvt) AS sd_s
        FROM '{train}' WHERE PHASE_mvt='DEP' GROUP BY 1 ORDER BY 2 DESC
    """).df().to_string())


if __name__ == "__main__":
    main()
