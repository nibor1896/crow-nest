#!/usr/bin/env bash
# #91 / #91: the TEACHER-FORCED K=2 measurement, engine side, one command.
#
# Fixes the exact sequence (serve's own render of the K=2 replay request, 6,738 prompt ids,
# + the bare container's greedy completion ids) and reads the raw top-20 distribution at
# EVERY completion position in each arm, via serve's `crow_force_ids` (the #81 injection
# door: each generated id is replaced by the forced one, logprobs price it under the model's
# distribution after the forced prefix). Same body bytes, render, prefill (chunk 2048),
# decode path and graphs as the live request - no re-tokenization anywhere.
#
#   arm        boot                                              what it answers
#   bare       no overlay            free greedy: THE ids (+ '1' at #65 must come back at -0.013)
#   bare-tf    same boot, forced     forcing is faithful (entries == the free run's)
#   dense      dense-bf16-originals  all non-expert weights BF16 originals (pinned 50 GiB)
#   dense-kv   + CROW_KV=bf16        ... and the KV cache BF16 (only expert FP4 left)
#   placebo    attn-v-out-control    kernel-path noise at #65 (numerically neutral overlay)
#
# Every boot needs the pinned pool back (#82): recover() balloons, and stops with
# "REBOOT needed" when MemAvailable stays flat below the arm's need. Arms already on disk
# are skipped, so a rerun after a reboot continues where it stopped.
#
# Usage (from anywhere; after a reboot is best):  tools/teacher-forced-91.sh [arm ...]
# Output: decode_out/91-teacher-forced/ (dumps, serve logs, compare.txt, oracle/gen-sequence.json)
set -uo pipefail
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAIN="${CROW_NEST_MAIN:-$HOME/Projects/crow-nest}"
cd "$WT" || exit 1
OUT=$WT/decode_out/91-teacher-forced
SD=$MAIN/decode_out/sessions/2026-09-22-diorama-rollover
mkdir -p "$OUT"
# the worktree runs its OWN serve (target/) against main's untracked data (container, overlays,
# hot-set sidecar, tokenizer): links, never copies
ln -sfn ../target engine/target
[ -e models ] || ln -sfn "$MAIN/models" models
for f in "$MAIN"/converter/*.cnq; do ln -sfn "$f" "converter/$(basename "$f")"; done
[ -e decode_out/hotsets-M-longctx2100-n160.json ] || ln -sfn "$MAIN/decode_out/hotsets-M-longctx2100-n160.json" decode_out/
[ -x target/release/serve ] || { echo "build first: cd engine && CARGO_TARGET_DIR=$WT/target cargo build --release --bin serve"; exit 2; }

avail() { awk '/^MemAvailable/{print int($2/1048576)}' /proc/meminfo; }
recover() {  # need_gib
  local need=$1 prev=-1 flat=0 av
  for i in 1 2 3 4 5 6 7 8; do
    av=$(avail); echo "recover: pass $i MemAvailable ${av} GiB (need $need)"
    [ "$av" -ge "$need" ] && return 0
    if [ "$prev" -ge 0 ] && [ "$av" -lt $((prev + 2)) ]; then flat=$((flat+1)); else flat=0; fi
    [ "$flat" -ge 2 ] && { echo "HARD: pool not coming back - REBOOT needed"; exit 3; }
    prev=$av
    timeout 300 python3 "$MAIN/decode_out/kv-ab/balloon.py" "$need" >/dev/null 2>&1
  done
  [ "$(avail)" -ge "$need" ] || { echo "pool did not reach $need GiB - REBOOT needed"; exit 3; }
}
SERVE=
boot() {  # label pinned_gb overlay [extra env...]
  local label=$1 pin=$2 ovl=$3; shift 3
  local envs=(env "CROW_RAM_MARGIN_GB=1" "CROW_PINNED_BUDGET_GB=$pin" "CROW_LOG=info,chat=debug" "CROW_CNQ_OVERLAY=" "$@")
  [ -n "$ovl" ] && envs+=("CROW_CNQ_OVERLAY=$ovl")
  "${envs[@]}" tools/serve-linux.sh --port 8099 > "$OUT/serve-$label.log" 2>&1 &
  SERVE=$!
  for _ in $(seq 1 150); do curl -sf http://127.0.0.1:8099/health >/dev/null && return 0; sleep 2; done
  echo "serve did not come up ($label) - see $OUT/serve-$label.log"; kill "$SERVE"; exit 2
}
stop() {
  kill "$SERVE" 2>/dev/null; wait "$SERVE" 2>/dev/null
  for _ in $(seq 1 120); do pgrep -f "target/release/serve --port 8099" >/dev/null || break; sleep 5; done
}
probe() {  # label [probe args...]
  local label=$1; shift
  python3 tools/corruption-replay-probe.py --session "$SD/session.json" --at 2 \
    --head-file tools/corpora/91-replay-diorama-0922-head.txt --crow-core "$SD/crow-3dbc015/cli/crow_core.py" \
    --rounds 1 --sampling '{"temperature":0}' --top-logprobs 20 --port 8099 \
    --label "$label" --json "$OUT/$label.json" --dump-lp "$OUT/$label.lp.json" "$@" \
    > "$OUT/probe-$label.log" 2>&1
  echo "probe $label exit $?"
}
IDS=$OUT/bare.lp.json
arm() {  # label need_gib pinned_gb overlay [extra env...]
  local label=$1 need=$2 pin=$3 ovl=$4; shift 4
  [ -s "$OUT/$label.lp.json" ] && { echo "=== $label: already on disk"; return 0; }
  [ -s "$IDS" ] || { [ "$label" = bare ] || { echo "=== $label: needs the bare ids first"; return 1; }; }
  echo "=== $(date +%T) $label"
  recover "$need"
  boot "$label" "$pin" "$ovl" "$@"
  if [ "$label" = bare ]; then
    probe bare                                   # free greedy: the ids of record
    probe bare-tf --force-ids "$IDS"             # the same ids forced (warm prefix restore)
  else
    probe "$label" --force-ids "$IDS"
  fi
  stop
}

ARMS=("$@"); [ ${#ARMS[@]} -gt 0 ] || ARMS=(bare dense dense-kv placebo)
for a in "${ARMS[@]}"; do
  case $a in
    bare)     arm bare 46 46 "" ;;
    dense)    arm dense 51 50 "$WT/converter/dense-bf16-originals.cnq" ;;
    dense-kv) arm dense-kv 51 50 "$WT/converter/dense-bf16-originals.cnq" CROW_KV=bf16 ;;
    placebo)  arm placebo 48 47 "$WT/converter/layer91-attn-v-out-control.cnq" ;;
    *) echo "unknown arm $a"; exit 2 ;;
  esac
done

cmp_args=()
for a in bare-tf dense dense-kv placebo; do [ -s "$OUT/$a.lp.json" ] && cmp_args+=(--arm "$a=$OUT/$a.lp.json"); done
python3 tools/teacher-forced-compare.py --ref "bare=$IDS" "${cmp_args[@]}" --at 65 --want 8 --got 1 \
  --json "$OUT/compare.json" | tee "$OUT/compare.txt"
# the oracle's input: serve's prompt ids (debug line of the bare boot) + the bare completion ids
python3 tools/teacher-forced-oracle-seq.py --serve-log "$OUT/serve-bare.log" --dump "$IDS" \
  --out "$OUT/oracle"
echo "=== $(date +%T) done"
