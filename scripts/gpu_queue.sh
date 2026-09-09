#!/usr/bin/env bash
# One card, one fit: N waiters on the same pid start N jobs at once, which is what made DQ1 measure a contended clock.
# A frozen trial used to take the whole queue with it (O2 trial 16, 2026-09-08: 2 h inside CatBoost's native trainer with the
# GPU at 13 W), so every job runs under a log-heartbeat watchdog and repeats until its declared study reaches k.
# The heartbeat is the trainer's own iteration output: trial boundaries alone killed the same healthy cell 3x on 2026-09-09.
set -u
export TAXI_CB_VERBOSE="${TAXI_CB_VERBOSE:-100}"  # CatBoost prints every 100 iterations, so the log mtime tracks the TRAINER
STALL_MIN="${STALL_MIN:-25}"   # a real heartbeat, so this is minutes of trainer silence, not minutes between trials
TRIES="${TRIES:-12}"
WATCH="${1:?winpid to wait for}"; shift

run_once() {  # module log -> runs it, killing the process if its log goes quiet for STALL_MIN
  local mod="$1" log="$2" pid age
  echo "start ${mod} $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${log}"
  python -u -m "${mod}" >> "${log}" 2>&1 &
  pid=$!
  while kill -0 "${pid}" 2>/dev/null; do
    sleep 60
    age=$(( $(date +%s) - $(stat -c %Y "${log}") ))
    if [ "${age}" -gt $(( STALL_MIN * 60 )) ]; then
      echo "stall ${mod} silent ${age}s, killed $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${log}"
      kill -9 "${pid}" 2>/dev/null
      break
    fi
  done
  wait "${pid}" 2>/dev/null
  echo "exit ${mod} rc=$? $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${log}"
}

while ps -W | grep -qE "[[:space:]]${WATCH}[[:space:]]"; do sleep 120; done
for JOB in "$@"; do
  MOD="${JOB%%:*}"; LOG="${JOB#*:}"
  for _ in $(seq 1 "${TRIES}"); do
    LEFT=$(PYTHONPATH=. python -m taxi.searches remaining "${MOD}")
    [ "${LEFT}" = "0" ] && break
    run_once "${MOD}" "${LOG}"
    [ "${LEFT}" = "-1" ] && break
  done
done
