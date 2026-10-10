#!/usr/bin/env bash
# Final population (final_ranking.parquet: Jan, Feb, Jun, Jul 2026): the frozen v53 recipe refit stage by stage.
# Each stage is skipped when its output exists, so a killed run resumes. Logs in data/final_chain/.
set -uo pipefail
cd "$(dirname "$0")/.."
L=data/final_chain
mkdir -p "$L"

run() {  # run OUT NAME CMD...: skip when OUT exists, fail loudly otherwise
  local out=$1 name=$2; shift 2
  if [ -f "$out" ]; then echo "$(date -u +%FT%TZ) skip $name"; return 0; fi
  echo "$(date -u +%FT%TZ) start $name"
  if "$@" > "$L/$name.log" 2>&1; then echo "$(date -u +%FT%TZ) done $name"; else echo "$(date -u +%FT%TZ) FAILED $name"; touch "$L/FAILED"; exit 1; fi
}
wait_for() { for f in "$@"; do until [ -f "$f" ]; do [ -f "$L/FAILED" ] && exit 1; sleep 30; done; done; }

gpu() {
  run data/mix_cb_wx_rank.parquet mix_cb_wx python mix.py fit mix_cb_wx 2309 3779
  run data/ra_cb_wz_rank.parquet ra_cb_wz python lanea.py fit ra_cb_wz mix_cb_wx 2373 1991 4509
  run data/nn3_wz_rank.parquet nn3_wz python nn.py fit nn3_wz mix_cb_wx
  run data/ra_cb_wz_t7_rank.parquet ra_cb_wz_t7 python lanea.py fit ra_cb_wz_t7 mix_cb_wx 2503 4354 6594
  wait_for data/cache/pw.parquet
  run data/nn3_pw_t7_rank.parquet nn3_pw_t7 python nn.py fit nn3_pw_t7 mix_cb_wx
}

cpu() {
  wait_for data/mix_cb_wx_rank.parquet
  run data/laneb_rank.parquet laneb python laneb.py fit 117 238
  run data/ra_lgb_wz_rank.parquet ra_lgb_wz python lanea.py fit ra_lgb_wz mix_cb_wx 229 286 494
  wait_for data/ra_cb_wz_rank.parquet data/nn3_wz_rank.parquet
  run data/cache/pw.parquet pw python -c "from features import pw_cols; pw_cols()"
  run data/ra_lgb_pw_t7_rank.parquet ra_lgb_pw_t7 python lanea.py fit ra_lgb_pw_t7 mix_cb_wx 241 7058 8248
}

gpu & G=$!
cpu & C=$!
wait $G; g=$?; wait $C; c=$?
[ $g -ne 0 ] || [ $c -ne 0 ] && { echo "chain failed (gpu $g, cpu $c)"; exit 1; }
# v53 = grid.py CLN: NNLS weights on OOF, then the second stage; lane B rows from laneb_rank
run data/recal_rank.parquet recal python recal.py fit ra_cb_wz_t7:0.442565 ra_lgb_pw_t7:0.168605 nn3_pw_t7:0.388830
echo "$(date -u +%FT%TZ) chain complete"
