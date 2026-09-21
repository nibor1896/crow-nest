#!/bin/bash
# #90 engine arm, the working invocation (2026-09-21):
#   - balloon before EVERY run (the pool refills after each engine exit)
#   - CROW_RAM_MARGIN_GB=1 and NO CROW_PINNED_BUDGET_GB override - the
#     planner keeps the default tier (the arm script's low ladder rungs
#     can never boot -M: low budget => more hot experts => VRAM refuses)
#   - the arm's own trim+hash steps per run (plan-rows.f32 is the marker;
#     after all four exist, tools/oracle_longctx_engine_arm.sh would just
#     skip them - this script does the same work directly)
# Usage: nohup bash tools/run-90-arm-final.sh > /tmp/90-arm-final.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
root=$PWD
OUT=decode_out/oracle-longctx/engine
PLAN=decode_out/oracle-longctx/row-plan.json
export LD_LIBRARY_PATH="$HOME/.local/share/crow/cuda/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

recover() {  # balloon until >=45 GiB, max 4 passes
  for i in 1 2 3 4; do
    local av; av=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
    [ "${av:-0}" -ge 45 ] && return 0
    echo "  balloon pass $i (avail ${av} GiB)"
    timeout 280 python3 "$root/decode_out/kv-ab/balloon.py" 46 >/dev/null 2>&1
  done
  awk '/MemAvailable/{print "  WARNING: avail only " int($2/1048576) " GiB"}' /proc/meminfo
}

run_one() {  # anchor arm
  local a=$1 arm=$2
  local ids=decode_out/oracle-longctx/longctx-170k-a${a}-ids.json
  local dir=$OUT/a${a}/${arm}
  [[ -f $dir/plan-rows.f32 ]] && { echo "anchor $a arm $arm: already done"; return 0; }
  mkdir -p "$dir"
  echo "== anchor $a arm $arm  ($(date +%T))"
  recover
  local envs=(CROW_CNQ=$root/converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq
              CROW_HOTSETS=$root/decode_out/hotsets-M-longctx2100-n160.json
              CROW_GRAPH=1 CROW_MMA=1 CROW_RAM_MARGIN_GB=1)
  [[ $arm == kvbf16 ]] && envs+=(CROW_KV=bf16 CROW_CHUNK=512)
  # ^ bf16 KV doubles the cache-side scratch; at anchor 2564 the auto chunk
  #   (2565) does not fit the 22 GiB VRAM budget (measured 12:00). The chunk
  #   cut is math-independent (gen.rs asserts it), so outputs stay identical.
  if ! env "${envs[@]}" engine/target/release/decode parity "$ids" "$dir" > "$dir/parity.log" 2>&1; then
    echo "anchor $a arm $arm: FAILED - see $dir/parity.log"; return 1
  fi
  local rows
  rows=$(python3 - "$PLAN" "$a" <<'EOF'
import json, sys
plan = json.load(open(sys.argv[1]))
g = [g for g in plan["groups"] if g["anchor"] == int(sys.argv[2])][0]
print(json.dumps(g["rows"]))
EOF
)
  python3 tools/oracle_longctx_rows.py subset --source "$dir/gpu-logits.f32" \
      --out "$dir/plan-rows.f32" --rows-file <(echo "$rows")
  sha256sum "$dir/gpu-logits.f32" "$dir/plan-rows.f32" "$ids" > "$dir/SHA256SUMS"
  rm -f "$dir/gpu-logits.f32"
  echo "anchor $a arm $arm: DONE (plan rows written, dense dump hashed+removed)"
}

fail=0
for a in 1000 2564; do
  for arm in none kvbf16; do
    run_one "$a" "$arm" || fail=1
  done
done
if [ $fail -eq 0 ]; then
  echo "ALL FOUR RUNS DONE - the #90 engine arm is complete. Log: this file."
else
  echo "SOME RUNS FAILED - rerun this script (idempotent; done runs are skipped)."
fi
