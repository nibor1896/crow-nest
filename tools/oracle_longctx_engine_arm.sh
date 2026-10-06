#!/usr/bin/env bash
# #90: the CROW-ENGINE arm of the long-context oracle form - teacher-forced
# `decode parity` dumps at the depth-plan anchors, wrapped in the GPU flock,
# trimmed to the plan's 64-row blocks, with a manifest per run.
#
# TWO FORMS OF DUMP. Up to anchor 2564 (the dense control at 1000 and the FIRST
# SPARSE rows at 2564) `decode parity` collects ALL rows of the anchor's ids file
# (engine/src/bin/decode.rs: "collecting all logits"): rows x 248,320 x 4 B, 1.0 GB
# at anchor 1000, 2.5 GB at 2564. That form is unchanged. Deeper anchors would be
# 50 GB at 50000, so they run in the TAIL form, CROW_PARITY_TAIL=<n> (decode.rs,
# docs/env.md): the ids before the plan's block are prefilled in 2,048-token chunks
# WITHOUT logits, then the block's n ids go teacher-forced through decode_step, one
# logits row each (plus 4 free decode rows). n is the row count of the anchor's
# plan group (64), accepted only when that group is one contiguous run ending at
# the last id of the anchor's prefix file - then the tail rows ARE the plan rows.
# The tail dump starts at the absolute row `row0_pos` (gen-sequence.json), which
# the trim hands to `oracle_longctx_rows.py subset --row0`. Tail rows come from the
# decode path, dense rows from the prefill path: pair arms WITHIN an anchor, and
# read a KLD-vs-position curve across the 2564/2565 line knowing the path changes.
#
# ARMS. --arms "none kvbf16" is the paired-baseline minimum: `none` is the
# default of record (the designated baseline), `kvbf16` the strictly-more-
# precision control that sizes the long-context noise floor the way §6 of
# docs/oracle-kld.md did at short context.
#
# --dry-run prints the `decode parity` command of every anchor x arm (first rung of
# the pinned-budget ladder) and exits: no flock, no engine, no GPU, nothing written.
#
# Usage:  tools/oracle_longctx_engine_arm.sh [--anchors "1000 2564"] [--arms "none kvbf16"] [--dry-run]
#         nohup tools/oracle_longctx_engine_arm.sh > decode_out/oracle-longctx/engine-arm.log 2>&1 &

set -euo pipefail
cd "$(dirname "$0")/.."

ANCHORS="1000 2564"
ARMS="none kvbf16"
DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --anchors) ANCHORS="$2"; shift 2 ;;
    --arms)    ARMS="$2";    shift 2 ;;
    --dry-run) DRY_RUN=1;    shift ;;
    -h|--help) sed -n '2,31p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown arg $1 (try --help)" >&2; exit 2 ;;
  esac
done

OUT=decode_out/oracle-longctx/engine
PLAN=decode_out/oracle-longctx/row-plan.json
CROW_CNQ=$PWD/converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq
HOTSETS=$PWD/decode_out/hotsets-M-longctx2100-n160.json
LD_LIB=$HOME/.local/share/crow/cuda/lib
# #131: the engine's own lock path (gen.rs engine_lock_path): CROW_LOCK=0 none, CROW_LOCK=<path>
# that path, else the per-user state dir - no longer engine/.engine.lock of the checkout
if [ "${CROW_LOCK:-}" = "0" ]; then lock=""
elif [ -n "${CROW_LOCK:-}" ]; then lock="$CROW_LOCK"
else lock="${XDG_STATE_HOME:-$HOME/.local/state}/crow-nest/engine.lock"; fi

# Anchors up to DENSE_MAX keep the dense form (every row collected); deeper ones run in the
# CROW_PARITY_TAIL form with n = the rows of the plan's block for that anchor
DENSE_MAX=2564
IDS_PREFIX=decode_out/oracle-longctx/longctx-170k-a
declare -A TAIL
tail_rows() {  # anchor: the plan block's row count, or an error when the block is not the prefix's tail
  python3 - "$PLAN" "$1" "${IDS_PREFIX}$1-ids.json" <<'EOF'
import json, sys
plan, anchor, ids_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
groups = [g for g in json.load(open(plan))["groups"] if g["anchor"] == anchor]
if not groups:
    sys.exit("anchor %d is not in %s" % (anchor, plan))
rows = groups[0]["rows"]
n_ids = len(json.load(open(ids_path)))
if not 0 < len(rows) < n_ids or rows != list(range(n_ids - len(rows), n_ids)):
    sys.exit("anchor %d: the plan's %d rows are not the last %d of the %d ids of %s"
             % (anchor, len(rows), len(rows), n_ids, ids_path))
print(len(rows))
EOF
}
for a in $ANCHORS; do
  if (( a > DENSE_MAX )); then
    TAIL[$a]=$(tail_rows "$a") || { echo "anchor $a: REFUSED - no tail form for it (see above)" >&2; exit 2; }
  fi
done
[[ -n "${ANCHORS//[[:space:]]/}" ]] || { echo "no runnable anchor left" >&2; exit 2; }

(( DRY_RUN )) || mkdir -p "$OUT"

parity_envs() {  # anchor arm pinned_gb: the env of one `decode parity` run, in the caller's `envs`
  local a=$1 arm=$2 pinned=$3
  envs=(CROW_CNQ=$CROW_CNQ CROW_HOTSETS=$HOTSETS CROW_GRAPH=1 CROW_MMA=1
        CROW_PINNED_BUDGET_GB=$pinned LD_LIBRARY_PATH=$LD_LIB)
  [[ $arm == kvbf16 ]] && envs+=(CROW_KV=bf16)
  [[ -n ${TAIL[$a]:-} ]] && envs+=(CROW_PARITY_TAIL=${TAIL[$a]})
  return 0
}

run_one() {  # anchor arm pinned_gb
  local a=$1 arm=$2 pinned=$3
  local ids=${IDS_PREFIX}${a}-ids.json
  local dir=$OUT/a${a}/${arm}
  [[ -f $dir/plan-rows.f32 ]] && { echo "anchor $a arm $arm: already done"; return 0; }
  # run_one is called under `if`, where set -e is off: every step that can fail says so itself
  mkdir -p "$dir" || return 1
  local envs
  parity_envs "$a" "$arm" "$pinned"
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
) || { echo "anchor $a arm $arm: no row-plan group - the dump $dir/gpu-logits.f32 is kept" >&2; return 1; }
  # the dense dump starts at row 0, the tail dump at the absolute row `row0_pos` of gen-sequence.json
  local row0=0 kind=dense
  if [[ -n ${TAIL[$a]:-} ]]; then
    kind=tail
    row0=$(python3 - "$dir/gen-sequence.json" "$rows" <<'EOF'
import json, sys
row0, first = json.load(open(sys.argv[1]))["row0_pos"], json.loads(sys.argv[2])[0]
if row0 != first:
    sys.exit("row0_pos %d is not the first plan row %d" % (row0, first))
print(row0)
EOF
) || { echo "anchor $a arm $arm: the tail dump does not start at the plan's first row" >&2; return 1; }
  fi
  # a failed trim or hash keeps the dump and leaves no plan-rows.f32: that file is the "done" marker
  python3 tools/oracle_longctx_rows.py subset --source "$dir/gpu-logits.f32" \
      --out "$dir/plan-rows.f32" --rows-file <(echo "$rows") --row0 "$row0" || {
    echo "anchor $a arm $arm: trim FAILED - the dump $dir/gpu-logits.f32 is kept" >&2
    rm -f "$dir/plan-rows.f32"; return 1; }
  sha256sum "$dir/gpu-logits.f32" "$dir/plan-rows.f32" "$ids" > "$dir/SHA256SUMS" || {
    echo "anchor $a arm $arm: hashing FAILED - the dump $dir/gpu-logits.f32 is kept" >&2
    rm -f "$dir/plan-rows.f32"; return 1; }
  rm -f "$dir/gpu-logits.f32" || return 1     # the plan rows are what the instrument reads
  echo "anchor $a arm $arm: plan rows written, $kind dump removed after hashing"
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

if (( DRY_RUN )); then   # what would run and nothing else: no lock, no engine, no GPU, no files
  for a in $ANCHORS; do
    for arm in $ARMS; do
      parity_envs "$a" "$arm" "${LADDER%% *}"
      if [[ -n ${TAIL[$a]:-} ]]; then form="tail form, CROW_PARITY_TAIL=${TAIL[$a]}"; else form="dense form, every row"; fi
      echo "dry-run anchor $a arm $arm ($form; first ladder rung ${LADDER%% *} GiB)"
      printf '  env'
      printf ' %q' "${envs[@]}" engine/target/release/decode parity "${IDS_PREFIX}${a}-ids.json" "$OUT/a${a}/${arm}"
      printf '\n'
    done
  done
  exit 0
fi

while (( $(pending) > 0 )) && (( $(date +%s) < DEADLINE )); do
  (
    exec 9>/tmp/crow-gpu.lock
    flock -w 1800 9 || { echo "no GPU flock inside 30 min - retrying later"; exit 0; }
    echo "gpu flock acquired $(date -Is)"
    while [[ -n $lock && -f $lock ]]; do
      if ! pgrep -f "engine/target/release/(serve|decode)" >/dev/null; then
        echo "$lock is stale (no engine process) - removing"
        rm -f "$lock"
      else
        echo "an engine is alive; waiting for $lock"; sleep 30
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
           "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           # the runs are dicts: sort by (anchor number, arm); a bare sorted() raises TypeError
           "runs": sorted(runs, key=lambda r: (r["anchor"].lstrip("a").zfill(8), r["arm"]))},
          open(os.path.join(out, "manifest.json"), "w"), indent=1)
print("manifest written: %d runs" % len(runs))
EOF
echo "engine arm complete $(date -Is)"
