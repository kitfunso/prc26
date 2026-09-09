#!/usr/bin/env bash
# One card, one fit: N waiters on the same pid start N jobs at once, which is what made DQ1 measure a contended clock.
# A frozen trial used to take the whole queue with it (O2 trial 16, 2026-09-08: 2 h inside CatBoost's native trainer with the
# GPU at 13 W), so every job runs under a log-heartbeat watchdog and repeats until its declared study reaches k.
# CatBoost's iteration output is C-buffered, so a healthy trial can go 25 min silent: the log alone killed 6 on 2026-09-09.
set -u
export TAXI_CB_VERBOSE="${TAXI_CB_VERBOSE:-100}"
STALL_MIN="${STALL_MIN:-25}"   # minutes the trainer must be BOTH silent and off the card before it counts as frozen
IDLE_PCT="${IDLE_PCT:-20}"
TRIES="${TRIES:-12}"
LOCKS="data/.queue-locks"
WATCH="${1:?winpid to wait for}"; shift

alive() { ps -W | grep -qE "[[:space:]]$1[[:space:]]"; }

claim() {  # module -> 0 if this supervisor now owns it. Three queues ran the same module at once on 2026-09-10.
  local mod="$1" owner
  mkdir -p "${LOCKS}"
  if mkdir "${LOCKS}/${mod}" 2>/dev/null; then echo $$ > "${LOCKS}/${mod}/pid"; return 0; fi
  owner=$(cat "${LOCKS}/${mod}/pid" 2>/dev/null || echo 0)
  if [ "${owner}" != "0" ] && kill -0 "${owner}" 2>/dev/null; then return 1; fi
  echo $$ > "${LOCKS}/${mod}/pid"
}

winpid_of() { ps -W | awk -v p="$1" '$1 == p { print $4 }'; }

run_once() {  # module log -> runs it, killing only a process that is silent AND off the card
  local mod="$1" log="$2" pid win age gpu
  echo "start ${mod} $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${log}"
  python -u -m "${mod}" >> "${log}" 2>&1 &
  pid=$!
  sleep 20; win=$(winpid_of "${pid}")
  while kill -0 "${pid}" 2>/dev/null; do
    sleep 60
    age=$(( $(date +%s) - $(stat -c %Y "${log}") ))
    [ "${age}" -le $(( STALL_MIN * 60 )) ] && continue
    gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ -n "${gpu}" ] && [ "${gpu}" -ge "${IDLE_PCT}" ] && continue
    echo "stall ${mod} silent ${age}s gpu ${gpu:-?}%, killing win ${win:-?} $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${log}"
    [ -n "${win}" ] && taskkill //F //T //PID "${win}" >> "${log}" 2>&1
    kill -9 "${pid}" 2>/dev/null
    break
  done
  wait "${pid}" 2>/dev/null
  echo "exit ${mod} rc=$? $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${log}"
  if [ -n "${win}" ] && alive "${win}"; then
    echo "ORPHAN ${mod} win ${win} survived the kill, refusing to relaunch" >> "${log}"
    return 1
  fi
}

while alive "${WATCH}"; do sleep 120; done
for JOB in "$@"; do
  MOD="${JOB%%:*}"; LOG="${JOB#*:}"
  claim "${MOD}" || { echo "skip ${MOD}: another queue holds the lock" >> "${LOG}"; continue; }
  for _ in $(seq 1 "${TRIES}"); do
    LEFT=$(PYTHONPATH=. python -m taxi.searches remaining "${MOD}")
    [ "${LEFT}" = "0" ] && break
    run_once "${MOD}" "${LOG}" || break
    [ "${LEFT}" = "-1" ] && break
  done
  rm -rf "${LOCKS:?}/${MOD}"
done
