#!/usr/bin/env bash
# #91 multi-site arm runner (2026-09-23). One serve boot per call, one arm per boot:
#   tools/multisite-0923.sh LABEL PINNED_GB OVERLAY|- [--freerun] [--speed] [ENV=VAL ...]
#
# What it runs: tools/multisite-corruption-probe.py `run` (and `freerun` / `speed` on request)
# against the site set tools/corpora/91-multisite-0923.json (23 corrupted tool-call sites of
# robin's 2026-09-23 diorama session). The site set references the session snapshots and the
# Crow c4f1b3c crow_core under decode_out/sessions/2026-09-23-diorama/ (untracked, sha-pinned
# in the site set; the probe refuses a different file).
#
# serve = $SERVE_ROOT/engine/target/release/serve, started through $SERVE_ROOT/tools/serve-linux.sh
# on port $PORT (default 8111; keep it off the port of a live engine). SERVE_ROOT defaults to this
# checkout; point it at another built tree to measure that tree's binary with this probe.
# Every boot is gated on `ramcheck --need PINNED_GB` (free_for_pin, #103). The engine log goes to
# $OUT/logs-LABEL (CROW_LOG_DIR), never to ~/.local/state/crow. The serve started here is stopped
# by its exact pid when the arm ends (trap).
#
# Environment: SERVE_ROOT, OUT (default decode_out/meas-multisite), PORT, CROW_CNQ / CROW_HOTSETS
# (default: the -M container and the longctx2100-n160 sidecar under SERVE_ROOT), ONLY=<site ids>,
# WARM=1 (--warm-chain). OVERLAY `-` = bare container.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVE_ROOT=${SERVE_ROOT:-$ROOT}
OUT=${OUT:-$ROOT/decode_out/meas-multisite}
PORT=${PORT:-8111}
SITES=$ROOT/tools/corpora/91-multisite-0923.json
PROBE=$ROOT/tools/multisite-corruption-probe.py
CNQ=${CROW_CNQ:-$SERVE_ROOT/converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq}
HOTSETS=${CROW_HOTSETS:-$SERVE_ROOT/decode_out/hotsets-M-longctx2100-n160.json}
[ $# -ge 3 ] || { sed -n 2,3p "$0"; exit 2; }
mkdir -p "$OUT"
label=$1 pin=$2 ovl=$3; shift 3
FREERUN=0 SPEED=0 ENVS=()
for a in "$@"; do
  case $a in --freerun) FREERUN=1;; --speed) SPEED=1;; *=*) ENVS+=("$a");; *) echo "bad arg $a"; exit 2;; esac
done
exec 9>"$OUT/.lock"; flock -n 9 || { echo "another multisite arm is running"; exit 1; }
if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; then echo "port $PORT is busy"; exit 2; fi

"$SERVE_ROOT/engine/target/release/ramcheck" --need "$pin" || { echo "ramcheck refused $pin GiB"; exit 3; }
envs=(env "CROW_CNQ=$CNQ" "CROW_HOTSETS=$HOTSETS"
      "CROW_RAM_MARGIN_GB=1" "CROW_PINNED_BUDGET_GB=$pin" "CROW_LOG=info"
      "CROW_LOG_DIR=$OUT/logs-$label" "CROW_CNQ_OVERLAY=none")
[ "$ovl" != "-" ] && envs+=("CROW_CNQ_OVERLAY=$ovl")
envs+=(${ENVS[@]+"${ENVS[@]}"})
echo "=== $(date +%T) boot $label: pinned $pin, overlay $ovl, env ${ENVS[*]:-none}"
( cd "$SERVE_ROOT" && exec "${envs[@]}" tools/serve-linux.sh --port "$PORT" ) > "$OUT/serve-$label.log" 2>&1 &
SERVE=$!
stop() {
  [ -n "${SERVE:-}" ] || return 0
  kill "$SERVE" 2>/dev/null
  for _ in $(seq 1 120); do kill -0 "$SERVE" 2>/dev/null || break; sleep 1; done
  kill -0 "$SERVE" 2>/dev/null && { echo "serve $SERVE still up after 120 s - SIGKILL"; kill -9 "$SERVE"; }
  wait "$SERVE" 2>/dev/null
  echo "=== $(date +%T) serve $SERVE ($label) stopped"
  SERVE=
}
trap stop EXIT
up=0
for _ in $(seq 1 300); do
  kill -0 "$SERVE" 2>/dev/null || break
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && { up=1; break; }
  sleep 2
done
[ $up = 1 ] || { echo "serve did not come up ($label) - see $OUT/serve-$label.log"; tail -5 "$OUT/serve-$label.log"; exit 2; }
echo "=== $(date +%T) serve up ($label, pid $SERVE)"
grep -o '"kv_dtype":"[^"]*"\|"residency_n":[0-9]*\|"pinned_bytes":[0-9]*' "$OUT/serve-$label.log" | tr '\n' ' '; echo
grep -h 'overlay' "$OUT/serve-$label.log" | head -3
grep -h '\[budget\] host pinned' "$OUT/serve-$label.log" | head -1

if [ $SPEED = 1 ]; then
  python3 "$PROBE" speed --port "$PORT" --label "$label" --json "$OUT/$label.speed.json" 2>>"$OUT/probe-$label.log"
fi
python3 "$PROBE" run --sites "$SITES" --port "$PORT" --label "$label" ${ONLY:+--only "$ONLY"} ${WARM:+--warm-chain} --json "$OUT/$label.json" 2>>"$OUT/probe-$label.log"
if [ $FREERUN = 1 ]; then
  python3 "$PROBE" freerun --sites "$SITES" --port "$PORT" --label "$label" --json "$OUT/$label.freerun.json" 2>>"$OUT/probe-$label.log"
fi
grep -h '\[chat\] prompt' "$OUT/serve-$label.log" | tail -3 | cut -c1-220
