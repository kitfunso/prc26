#!/usr/bin/env bash
# The runner reads JOBS at import, so a job added after it started needs a fresh process; markers make the relaunch idempotent.
set -u
WATCH="${1:?winpid of the running queue}"
ps -W | grep -qE "[[:space:]]${WATCH}[[:space:]]" || { echo "pid ${WATCH} not running, refusing to relaunch"; exit 1; }
while ps -W | grep -qE "[[:space:]]${WATCH}[[:space:]]"; do sleep 60; done
echo "queue ${WATCH} exited $(date), relaunching"
exec python scripts/run_queue.py
