"""One-off measurements for the protocol review: scored-subset candidates, AOBT_3-missing rows, tail SSE share."""
import duckdb

D = "data"
TR = f"read_parquet('{D}/training_*.parquet')"
RK = f"read_parquet('{D}/ranking.parquet')"


def show(title: str, sql: str) -> None:
    print(f"\n== {title}")
    print(duckdb.sql(sql).df().to_string(index=False))


show("training DEP rows per month: total vs with target vs with AOBT_3", f"""
SELECT strftime(MVT_TIME_UTC_mvt, '%Y-%m') m, count(*) dep_rows, count(TAXITIME_SEC_mvt) with_target,
       round(count(TAXITIME_SEC_mvt) * 1.0 / count(*), 4) frac_target, count(AOBT_3_flt) with_aobt3,
       count(BLOCK_TIME_UTC_mvt) with_block
FROM {TR} WHERE PHASE_mvt = 'DEP' GROUP BY 1 ORDER BY 1""")

show("training DEP totals", f"""
SELECT count(*) dep_rows, count(TAXITIME_SEC_mvt) with_target, round(count(TAXITIME_SEC_mvt) * 1.0 / count(*), 4) frac_target
FROM {TR} WHERE PHASE_mvt = 'DEP'""")

show("ranking DEP totals", f"""
SELECT count(*) n, count(AOBT_3_flt) with_aobt3, count(TAXITIME_SEC_mvt) with_target, count(BLOCK_TIME_UTC_mvt) with_block
FROM {RK} WHERE PHASE_mvt = 'DEP'""")

show("submitting.parquet shape", f"SELECT count(*) n, count(DISTINCT MVT_ID_mvt) ids FROM read_parquet('{D}/submitting.parquet')")
print(duckdb.sql(f"DESCRIBE SELECT * FROM read_parquet('{D}/submitting.parquet')").df().to_string(index=False))

show("scored-count arithmetic: 215876 vs candidates", f"""
SELECT 215876 AS used_pairs,
       (SELECT count(*) FROM {RK} WHERE PHASE_mvt = 'DEP') AS rank_dep,
       (SELECT count(*) FROM read_parquet('{D}/submitting.parquet')) AS submitting_rows,
       round(215876.0 / (SELECT count(*) FROM {RK} WHERE PHASE_mvt = 'DEP'), 4) AS frac_of_rank_dep""")

show("training: rows without AOBT_3 vs with, target stats", f"""
SELECT AOBT_3_flt IS NULL AS no_aobt3, count(*) n, round(avg(TAXITIME_SEC_mvt)) mean_t,
       round(sqrt(avg(TAXITIME_SEC_mvt::DOUBLE * TAXITIME_SEC_mvt))) rms_t, round(quantile_cont(TAXITIME_SEC_mvt, 0.5)) med_t
FROM {TR} WHERE PHASE_mvt = 'DEP' AND TAXITIME_SEC_mvt IS NOT NULL GROUP BY 1""")

show("training: tail buckets and their share of SSE around the per-airport mean", f"""
WITH t AS (SELECT ADEP_mvt a, TAXITIME_SEC_mvt y FROM {TR} WHERE PHASE_mvt = 'DEP' AND TAXITIME_SEC_mvt IS NOT NULL),
m AS (SELECT a, avg(y) mu FROM t GROUP BY a),
j AS (SELECT y, mu FROM t JOIN m USING (a))
SELECT CASE WHEN y <= 0 THEN 'nonpos' WHEN y > 14400 THEN 'over 4h' WHEN y > 7200 THEN '2h-4h'
            WHEN y > 3600 THEN '1h-2h' ELSE 'up to 1h' END bucket,
       count(*) n, round(100.0 * sum((y - mu) * (y - mu)) / (SELECT sum((y - mu) * (y - mu)) FROM j), 2) pct_sse
FROM j GROUP BY 1 ORDER BY 2 DESC""")

show("holdout months: same tail table on Jan+Jul 2025", f"""
WITH t AS (SELECT ADEP_mvt a, TAXITIME_SEC_mvt y FROM {TR}
           WHERE PHASE_mvt = 'DEP' AND TAXITIME_SEC_mvt IS NOT NULL
             AND (MVT_TIME_UTC_mvt < '2025-02-01' OR (MVT_TIME_UTC_mvt >= '2025-07-01' AND MVT_TIME_UTC_mvt < '2025-08-01'))),
m AS (SELECT a, avg(y) mu FROM t GROUP BY a),
j AS (SELECT y, mu FROM t JOIN m USING (a))
SELECT CASE WHEN y <= 0 THEN 'nonpos' WHEN y > 14400 THEN 'over 4h' WHEN y > 7200 THEN '2h-4h'
            WHEN y > 3600 THEN '1h-2h' ELSE 'up to 1h' END bucket,
       count(*) n, round(100.0 * sum((y - mu) * (y - mu)) / (SELECT sum((y - mu) * (y - mu)) FROM j), 2) pct_sse
FROM j GROUP BY 1 ORDER BY 2 DESC""")
