#!/bin/bash
# perf-loop iteration: build → quality gates → benchmark → history row.
# Usage: tools/perf_loop.sh <label> [extra env as KEY=VAL ...]
# Every run appends one line to decode_out/perf_history.tsv so each
# improvement is measured against the previous — the loop runs until a full
# round brings no measurable gain (ceiling reached).
LABEL="$1"; shift
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/engine" || exit 1

echo "== perf-loop [$LABEL] build =="
BUILD_ERR=$(cargo build --release --bins 2>&1 | grep -cE "^error")
if [ "$BUILD_ERR" != "0" ]; then echo "BUILD_FAILED"; exit 1; fi

OUT="$ROOT/decode_out/perfloop-$LABEL"
mkdir -p "$OUT"

echo "== gate: layercheck (bound 0.125) =="
CROW_CNQ=../converter/Qwen3.8-Flash-Next-CNQ4.5.cnq env "$@" target/release/decode.exe layercheck 2>/dev/null | tail -1 | tee "$OUT/lc.txt"

echo "== bench: 12-position argmax =="
CROW_CNQ=../converter/Qwen3.8-Flash-Next-CNQ4.5.cnq env "$@" target/release/decode.exe parity ../decode_out/demo-ids.json "$OUT" > "$OUT/parity.log" 2>&1
tail -1 "$OUT/parity.log"

echo "== bench: decode 64 steps =="
CROW_CNQ=../converter/Qwen3.8-Flash-Next-CNQ4.5.cnq env "$@" target/release/decode.exe run ../decode_out/demo-ids.json 64 > "$OUT/run64.log" 2>&1
grep "decode:" "$OUT/run64.log" | tee "$OUT/ms.txt"

echo "== bench: prefill rate (large prompt = t7 material head) =="
head -c 40000 "$ROOT/decode_out/ten-tasks-sorted.json" > /dev/null  # file exists check
CROW_CNQ=../converter/Qwen3.8-Flash-Next-CNQ4.5.cnq env "$@" target/release/decode.exe run ../decode_out/prefill-bench-ids.json 4 > "$OUT/prefill.log" 2>&1
grep -E "prefill done|decode:" "$OUT/prefill.log" | tee -a "$OUT/ms.txt"

# history row
TS=$(date +"%Y-%m-%d %H:%M")
LCV=$(grep -oE "max_abs=[0-9.e+-]+" "$OUT/lc.txt" | head -1)
MS=$(grep -oE "mean [0-9.]+ ms" "$OUT/run64.log" | head -1)
PF=$(grep -oE "[0-9.]+ tok/s completed-prompt average" "$OUT/prefill.log" | head -1)
echo -e "$TS\t$LABEL\t$LCV\t$MS\t$PF\t$*" >> "$ROOT/decode_out/perf_history.tsv"
echo "== history row appended =="
