#!/usr/bin/env bash
# Serialise a GPU job behind one already running: 16 GB holds the two fits now on the card, not a third.
set -u
WATCH="${1:?winpid to wait for}"; MOD="${2:?module}"; LOG="${3:?log file}"
while ps -W | grep -qE "[[:space:]]${WATCH}[[:space:]]"; do sleep 120; done
echo "pid ${WATCH} gone $(date -u +%Y-%m-%dT%H:%M:%SZ), starting ${MOD}" >> "${LOG}"
exec python -u -m "${MOD}" >> "${LOG}" 2>&1
