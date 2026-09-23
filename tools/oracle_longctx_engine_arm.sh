#!/usr/bin/env bash
# #90: the CROW-ENGINE arm of the long-context oracle form - teacher-forced
# `decode parity` dumps at the depth-plan anchors, wrapped in the GPU flock,
# trimmed to the plan's 64-row blocks, with a manifest per run.
#
# WHAT RUNS TONIGHT AND WHAT CANNOT. `decode parity` collects ALL rows of its
# ids file (engine/src/bin/decode.rs: "collecting all logits"), so an anchor's
# dump is rows x 248,320 x 4 B: 1.0 GB at anchor 1000, 2.5 GB at 2564, and
# 50 GB at 50000 - the first two are runnable, the rest are blocked by that
# collect-all design (a --rows/stride flag in parity is the fix; engine/ is not
# this issue's to touch). This script therefore runs the anchors it can
# (--anchors, default "1000 2564") and refuses the rest with the reason.
#
# ARMS. --arms "none kvbf16" is the paired-baseline minimum: `none` is the
# default of record (the designated baseline), `kvbf16` the strictly-more-
# precision control that sizes the long-context noise floor the way §6 of
# docs/oracle-kld.md did at short context.
#
# Usage:  tools/oracle_longctx_engine_arm.sh [--anchors "1000 2564"] [--arms "none kvbf16"]
#         nohup tools/oracle_longctx_engine_arm.sh > decode_out/oracle-longctx/engine-arm.log 2>&1 &

set -euo pipefail
cd "$(dirname "$0")/.."

ANCHORS="1000 2564"
ARMS="none kvbf16"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --anchors) ANCHORS="$2"; shift 2 ;;
    --arms)    ARMS="$2";    shift 2 ;;
    -h|--help) sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown arg $1 (try --help)" >&2; exit 2 ;;
  esac
done

OUT=decode_out/oracle-longctx/engine
PLAN=decode_out/oracle-longctx/row-plan.json
CROW_CNQ=$PWD/converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq
HOTSETS=$PWD/decode_out/hotsets-M-longctx2100-n160.json
LD_LIB=$HOME/.local/share/crow/cuda/lib

for a in $ANCHORS; do
  if (( a > 2564 )); then
    echo "anchor $a: REFUSED - decode parity collects every row, $a rows would be \
$(( a * 248320 * 4 / 1000000000 )) GB of logits; needs a --rows flag in parity first" >&2
  fi
done
ANCHORS=$(for a in $ANCHORS; do (( a <= 2564 )) && echo $a; done)
[[ -n "$ANCHORS" ]] || { echo "no runnable anchor left" >&2; exit 2; }

mkdir -p "$OUT"

run_one() {  # anchor arm pinned_gb
  local a=$1 arm=$2 pinned=$3
  local ids=decode_out/oracle-longctx/longctx-170k-a${a}-ids.json
  local dir=$OUT/a${a}/${arm}
  [[ -f $dir/plan-rows.f32 ]] && { echo "anchor $a arm $arm: already done"; return 0; }
  mkdir -p "$dir"
  local envs=(CROW_CNQ=$CROW_CNQ CROW_HOTSETS=$HOTSETS CROW_GRAPH=1 CROW_MMA=1
              CROW_PINNED_BUDGET_GB=$pinned LD_LIBRARY_PATH=$LD_LIB)
  [[ $arm == kvbf16 ]] && envs+=(CROW_KV=bf16)
  echo "== anchor $a arm $arm pinned ${pinned} GiB  ($(date -Is))"
  env "${envs[@]}" engine/target/release/decode parity "$ids" "$dir" \
      > "$dir/parity.log" 2>&1 || { echo "anchor $a arm $arm: parity FAILED - see $dir/parity.log" >&2; return 1; }
  grep -E "decode/parity|prompt tokens|prefill" "$dir/parity.log" || true
  [[ -f $dir/gpu-logits.f32 ]] || { echo "anchor $a arm $arm: no dump written" >&2; return 1; }
  # trim the dense dump to the plan's rows for this anchor and record both hashes
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
  rm -f "$dir/gpu-logits.f32"     # the plan rows are what the instrument reads
  echo "anchor $a arm $arm: plan rows written, dense dump removed after hashing"
}

# The fleet shares this machine: when host RAM is squeezed the loader refuses
# its config (manager.rs:203). Retry a ladder of pinned budgets with backoff
# until every run is done or the deadline passes - the runs are idempotent.
# The GPU flock is held ONLY during an attempt round, never during the backoff
# sleep, so other flock-cooperating agents are not blocked by our waiting.
DEADLINE=$(( $(date +%s) + 8 * 3600 ))
LADDER="${PINNED_LADDER:-50 36 24 16}"
pending() {
  local todo=0
  for a in $ANCHORS; do for arm in $ARMS; do
    [[ -f $OUT/a${a}/${arm}/plan-rows.f32 ]] || todo=$((todo + 1))
  done; done
  echo $todo
}

while (( $(pending) > 0 )) && (( $(date +%s) < DEADLINE )); do
  (
    exec 9>/tmp/crow-gpu.lock
    flock -w 1800 9 || { echo "no GPU flock inside 30 min - retrying later"; exit 0; }
    echo "gpu flock acquired $(date -Is)"
    while [[ -f engine/.engine.lock ]]; do
      if ! pgrep -f "engine/target/release/(serve|decode)" >/dev/null; then
        echo "engine/.engine.lock is stale (no engine process) - removing"
        rm -f engine/.engine.lock
      else
        echo "an engine is alive; waiting for engine/.engine.lock"; sleep 30
      fi
    done
    for a in $ANCHORS; do
      for arm in $ARMS; do
        [[ -f $OUT/a${a}/${arm}/plan-rows.f32 ]] && continue
        for pinned in $LADDER; do
          if run_one "$a" "$arm" "$pinned"; then break; fi
          grep -q "refusing config\|refusing to pin" "$OUT/a${a}/${arm}/parity.log" \
            || break    # a real failure, not the RAM/VRAM squeeze - stop this arm
        done
      done
    done
  )
  if (( $(pending) > 0 )); then
    echo "$(pending) run(s) still missing; backing off 10 min (fleet RAM/VRAM squeeze)"
    sleep 600
  fi
done

todo=$(pending)
if (( todo > 0 )); then
  echo "engine arm: $todo run(s) NOT done - the machine never freed enough RAM/VRAM" \
       "before the deadline; the commands are in this script and the docs" >&2
fi

python3 - <<'EOF'
import json, os, time
out = "decode_out/oracle-longctx/engine"
runs = []
for root, dirs, files in os.walk(out):
    if "parity.log" in files:
        arm = os.path.basename(root)
        anchor = os.path.basename(os.path.dirname(root))
        runs.append({"anchor": anchor, "arm": arm,
                     "log": os.path.relpath(os.path.join(root, "parity.log"), out),
                     "done": os.path.exists(os.path.join(root, "plan-rows.f32"))})
json.dump({"what": "#90 crow-engine longctx arm runs (paired-baseline mode)",
           "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "runs": sorted(runs)},
          open(os.path.join(out, "manifest.json"), "w"), indent=1)
print("manifest written: %d runs" % len(runs))
EOF
echo "engine arm complete $(date -Is)"
