"""Are the huge-taxi rows visible through the AOBT_3 gap? Holdout Jan+Jul 2025, DEP rows with AOBT_3."""
import duckdb

D = "data"
BASE = f"""
SELECT ADEP_mvt a, TAXITIME_SEC_mvt::DOUBLE y,
       epoch(MVT_TIME_UTC_mvt) - epoch(AOBT_3_flt) AS gap
FROM read_parquet('{D}/training_*.parquet')
WHERE PHASE_mvt = 'DEP' AND TAXITIME_SEC_mvt IS NOT NULL AND AOBT_3_flt IS NOT NULL
  AND (MVT_TIME_UTC_mvt < '2025-02-01' OR (MVT_TIME_UTC_mvt >= '2025-07-01' AND MVT_TIME_UTC_mvt < '2025-08-01'))"""

print(duckdb.sql(f"""
WITH t AS ({BASE}), r AS (SELECT y, gap, gap - y AS res FROM t)
SELECT CASE WHEN y <= 0 THEN 'nonpos' WHEN y > 14400 THEN 'over 4h' WHEN y > 7200 THEN '2h-4h'
            WHEN y > 3600 THEN '1h-2h' ELSE 'up to 1h' END bucket,
       count(*) n, round(avg(y)) mean_y, round(avg(gap)) mean_gap, round(avg(res)) mean_res,
       round(sqrt(avg(res * res))) rms_res,
       round(100.0 * sum(res * res) / (SELECT sum(res * res) FROM r), 2) pct_res_sse,
       sum(CASE WHEN abs(res) > 600 THEN 1 ELSE 0 END) n_res_over_600
FROM r GROUP BY 1 ORDER BY 2 DESC""").df().to_string(index=False))

print("\n== residual tail by gap bucket (is a big gap a big residual?)")
print(duckdb.sql(f"""
WITH t AS ({BASE}), r AS (SELECT y, gap, gap - y AS res FROM t)
SELECT CASE WHEN gap > 14400 THEN 'gap over 4h' WHEN gap > 7200 THEN 'gap 2h-4h' WHEN gap > 3600 THEN 'gap 1h-2h'
            WHEN gap < 0 THEN 'gap negative' ELSE 'gap up to 1h' END bucket,
       count(*) n, round(avg(y)) mean_y, round(avg(res)) mean_res, round(sqrt(avg(res * res))) rms_res,
       round(100.0 * sum(res * res) / (SELECT sum(res * res) FROM r), 2) pct_res_sse
FROM r GROUP BY 1 ORDER BY 2 DESC""").df().to_string(index=False))

print("\n== top 12 residual rows")
print(duckdb.sql(f"""
WITH t AS ({BASE}) SELECT a, y, gap, gap - y AS res FROM t ORDER BY abs(gap - y) DESC LIMIT 12""").df().to_string(index=False))
