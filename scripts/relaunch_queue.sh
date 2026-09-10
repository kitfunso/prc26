#!/usr/bin/env bash
# The runner reads JOBS at import, so a job added after it started needs a fresh process; markers make the relaunch idempotent.
# Trailing args are job names, passed through to run_queue.py, so a chain can wait on any GPU pid and then run a named subset.
set -u
WATCH="${1:?winpid to wait for}"; shift
ps -W | grep -qE "[[:space:]]${WATCH}[[:space:]]" || { echo "pid ${WATCH} not running, refusing to relaunch"; exit 1; }
while ps -W | grep -qE "[[:space:]]${WATCH}[[:space:]]"; do sleep 60; done
echo "pid ${WATCH} exited $(date -u +%Y-%m-%dT%H:%M:%SZ), starting queue ${*:-all}"
exec python scripts/run_queue.py "$@"
