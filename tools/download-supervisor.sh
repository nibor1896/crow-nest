#!/bin/bash
# download-supervisor.sh — crow-nest #5 prerequisite: keep the 360 GB original-safetensors
# download alive. Huggingface drops transfers silently; hf resumes from its cache, so the
# supervisor simply restarts on exit or stall and logs every action.
#
# Checks every CHECK_SEC: process alive? size growing? Restarts on: process dead before
# completion, or no growth for STALL_SEC. Completes when size >= EXPECTED_GB.

DEST="/c/Users/robin/dev/crow-nest/models/Qwen3.8-Flash-Next-original"
SVLOG="/c/Users/robin/dev/crow-nest/models/download-supervisor.log"
HFLOG="/c/Users/robin/dev/crow-nest/models/download-hf.log"
REV="de4b8e4d43b917e7706784d8bb445c9af86a3540"
EXPECTED_BYTES=359999963128   # exact safetensors total from the model index
EXPECTED_COUNT=131            # shard count from the model index
CHECK_SEC=45
STALL_SEC=240            # no growth for 10 min -> restart
MAX_BACKOFF=300

say() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> "$SVLOG"; }

size_gb() { du -sB1G "$DEST" 2>/dev/null | cut -f1; }
# REAL completion: count finished shards and their exact bytes — du lies while
# chunk tempfiles exist (it declared COMPLETE ~20 GB early once; rule: verify at object).
done_bytes() { find "$DEST" -name '*.safetensors' -printf '%s
' 2>/dev/null | awk '{s+=$1} END {print s+0}'; }
done_count() { find "$DEST" -name '*.safetensors' 2>/dev/null | wc -l; }

start_dl() {
  HF_HUB_DISABLE_XET=1 hf download Qwen/Qwen3.8-Flash-Next \
    --revision "$REV" --local-dir "$DEST" >> "$HFLOG" 2>&1 &
  DL_PID=$!
  LAST_SIZE=$(size_gb); LAST_GROWTH=$(date +%s)
  BACKOFF=5
  say "started hf (pid $DL_PID) at ${LAST_SIZE:-0} GB"
}

say "supervisor start (expect ${EXPECTED_GB} GB in $DEST)"
start_dl
while true; do
  sleep "$CHECK_SEC"
  SIZE=$(size_gb)
  NOW=$(date +%s)
  BYTES=$(done_bytes); COUNT=$(done_count)
  if [ "$BYTES" -ge "$EXPECTED_BYTES" ] && [ "$COUNT" -ge "$EXPECTED_COUNT" ]; then
    say "COMPLETE: $COUNT shards, $BYTES bytes (exact)"
    exit 0
  fi
  if ! kill -0 "$DL_PID" 2>/dev/null; then
    say "hf died (pid $DL_PID) at ${SIZE:-?} GB -> restart in ${BACKOFF}s"
    sleep "$BACKOFF"; BACKOFF=$(( BACKOFF * 2 )); [ "$BACKOFF" -gt "$MAX_BACKOFF" ] && BACKOFF=$MAX_BACKOFF
    start_dl
    continue
  fi
  if [ -n "$SIZE" ] && [ "$SIZE" != "$LAST_SIZE" ]; then
    LAST_SIZE=$SIZE; LAST_GROWTH=$NOW; BACKOFF=5
  elif [ $(( NOW - LAST_GROWTH )) -ge "$STALL_SEC" ]; then
    say "STALL: no growth for $STALL_SEC s at ${SIZE:-?} GB -> kill pid $DL_PID and restart"
    kill "$DL_PID" 2>/dev/null; sleep 5
    kill -9 "$DL_PID" 2>/dev/null
    start_dl
  fi
done
