"""Pull the summary tables that back the data-viewer page into data/view.json."""
import json
import duckdb

TRAIN = "data/training_*.parquet"
RANK = "data/ranking.parquet"
con = duckdb.connect()
con.execute(f"CREATE VIEW t AS SELECT * FROM read_parquet('{TRAIN}', filename=true)")
con.execute(f"CREATE VIEW r AS SELECT * FROM '{RANK}'")
con.execute("CREATE VIEW d AS SELECT * FROM t WHERE PHASE_mvt='DEP' AND TAXITIME_SEC_mvt IS NOT NULL")
out: dict = {}

def q(sql: str) -> list[dict]:
    return json.loads(con.sql(sql).df().to_json(orient="records", date_format="iso"))

out["months"] = q("""
  SELECT regexp_extract(filename, 'training_([0-9]{4}-[0-9]{2})', 1) AS month,
         count(*) AS rows, count(*) FILTER (WHERE PHASE_mvt='DEP') AS dep,
         count(*) FILTER (WHERE PHASE_mvt='ARR') AS arr
  FROM t GROUP BY 1 ORDER BY 1""")
out["ranking"] = q("""
  SELECT strftime(MVT_TIME_UTC_mvt, '%Y-%m') AS month, count(*) AS rows,
         count(*) FILTER (WHERE PHASE_mvt='DEP') AS dep,
         count(*) FILTER (WHERE PHASE_mvt='ARR') AS arr
  FROM r GROUP BY 1 ORDER BY 1""")

cols = [c[0] for c in con.sql("DESCRIBE t").fetchall() if c[0] != "filename"]
types = {c[0]: c[1] for c in con.sql("DESCRIBE t").fetchall()}
cat = []
for c in cols:
    s = con.sql(f"""SELECT count({c}) AS nn, count(DISTINCT {c}) AS nd FROM d""").fetchone()
    rk = con.sql(f"SELECT count({c}) FROM r WHERE PHASE_mvt='DEP'").fetchone()[0]
    ex = [str(v[0]) for v in con.sql(
        f"SELECT {c} FROM d WHERE {c} IS NOT NULL USING SAMPLE 3 ROWS").fetchall()]
    cat.append({"col": c, "type": types[c], "nonnull_train": s[0], "distinct": s[1],
                "nonnull_rank": rk, "examples": ex})
n_dep_train = con.sql("SELECT count(*) FROM d").fetchone()[0]
n_dep_rank = con.sql("SELECT count(*) FROM r WHERE PHASE_mvt='DEP'").fetchone()[0]
out["catalogue"] = cat
out["n_dep_train"] = n_dep_train
out["n_dep_rank"] = n_dep_rank

out["sample"] = q(f"""
  SELECT {', '.join(cols)} FROM d
  WHERE ADEP_mvt='EGLL' AND MVT_TIME_UTC_mvt >= '2025-07-15 06:00:00+00'
  ORDER BY MVT_TIME_UTC_mvt LIMIT 24""")

out["airports"] = q("""
  SELECT ADEP_mvt AS apt, count(*) AS n, round(avg(TAXITIME_SEC_mvt)) AS mean,
         quantile_cont(TAXITIME_SEC_mvt, 0.1) AS p10, median(TAXITIME_SEC_mvt) AS p50,
         quantile_cont(TAXITIME_SEC_mvt, 0.9) AS p90, round(stddev(TAXITIME_SEC_mvt)) AS sd,
         max(TAXITIME_SEC_mvt) AS mx, count(*) FILTER (WHERE TAXITIME_SEC_mvt > 3600) AS over1h,
         count(*) FILTER (WHERE TAXITIME_SEC_mvt <= 0) AS nonpos
  FROM d GROUP BY 1 ORDER BY 2 DESC""")

out["hist"] = q("""
  SELECT ADEP_mvt AS apt, least(floor(TAXITIME_SEC_mvt/60), 45)::INT AS m, count(*) AS n
  FROM d GROUP BY 1,2 ORDER BY 1,2""")

out["hour"] = q("""
  SELECT ADEP_mvt AS apt, hour(MVT_TIME_UTC_mvt) AS h, median(TAXITIME_SEC_mvt) AS p50, count(*) AS n
  FROM d GROUP BY 1,2 ORDER BY 1,2""")

out["leak"] = q("""
  WITH e AS (SELECT epoch(MVT_TIME_UTC_mvt) - epoch(AOBT_3_flt) - TAXITIME_SEC_mvt AS err
             FROM d WHERE AOBT_3_flt IS NOT NULL)
  SELECT greatest(least(floor(err/30), 30), -30)::INT AS b, count(*) AS n FROM e GROUP BY 1 ORDER BY 1""")
out["leak_stats"] = q("""
  WITH e AS (SELECT epoch(MVT_TIME_UTC_mvt) - epoch(AOBT_3_flt) - TAXITIME_SEC_mvt AS err
             FROM d WHERE AOBT_3_flt IS NOT NULL)
  SELECT count(*) AS n, round(avg(err),1) AS bias, round(sqrt(avg(err*err)),1) AS rmse,
         median(abs(err)) AS mad, round(100.0*count(*) FILTER (WHERE abs(err)<=60)/count(*),1) AS pct_within_60s,
         round(100.0*count(*) FILTER (WHERE abs(err)>600)/count(*),1) AS pct_over_10min
  FROM e""")

json.dump(out, open("data/view.json", "w"), indent=0)
print("months", len(out["months"]), "| dep train", n_dep_train, "| dep rank", n_dep_rank)
print(json.dumps(out["airports"], indent=0)[:1500])
print(json.dumps(out["leak_stats"]))

tmpl = open("docs/data-viewer.tmpl.html", encoding="utf-8").read()
blob = json.dumps(out, separators=(",", ":")).replace("</", "<" + chr(92) + "/")
open("docs/data-viewer.html", "w", encoding="utf-8").write(tmpl.replace("/*DATA*/", blob))
