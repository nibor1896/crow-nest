#!/usr/bin/env bash
# #91 corruption arms -- the decisive measurement, run end to end.
#
# For each arm: boot serve with the arm's overlay (CROW_CNQ_OVERLAY), wait for
# /health, run tools/corruption-probe.py (8 seeded rounds, 40 hex literals per
# round), stop serve, wait for the pinned pool to come back, next arm. Results
# land in decode_out/corruption-arms/<label>.json plus a summary table.
#
# The arms (converter/layer_rule_overlay.rs, #91 phase 1):
#   baseline    no overlay -- the shipped CNQ4.5-M
#   attn-ctrl   attn-v-out-control    (byte-identical placebo; proves the mechanism)
#   attn-arm    attn-v-out-originals  (v/o/linear-out projections -> BF16, all layers)
#   rule-arm    ffn-down-rule-originals (shared-expert down_proj -> BF16, llama.cpp's
#                                        use_more_bits layer set)
#   all-arm     ffn-down-all-originals  (same tensor, ALL layers)
set -uo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# CORRUPTION_CTX_TOKENS=100000 runs the ladder with the LONG-CONTEXT probe
# (tools/corruption-probe-long.py: same literals, same grader, N tokens of
# session in front; CORRUPTION_POSITION=end|start). The short probe sat on its
# floor at 984 prompt tokens (2026-09-21, all arms ~0.003); results of a depth
# get their own directory, so the skip guard and the summary never mix depths.
ctx_tokens="${CORRUPTION_CTX_TOKENS:-}"
position="${CORRUPTION_POSITION:-end}"
# CORRUPTION_SAMPLING: JSON merged over the probe's sampling (long probe only),
# CORRUPTION_TAG: names the result directory of such a variant, so it never
# shares a directory (skip guard, summary) with the record.
sampling="${CORRUPTION_SAMPLING:-}"
# CORRUPTION_REPLAY=<preset> runs the ladder with the REPLAY probe
# (tools/corruption-replay-probe.py: a stored Crow session cut at the answers
# that came back corrupt live, the body Crow builds for it, every returned tool
# call graded); CORRUPTION_REPLAY_SESSION names the session SNAPSHOT (the probe
# refuses a file whose sha256 is not the preset's). Its own result directory,
# like a depth: the numbers are calls, not literal lines.
replay="${CORRUPTION_REPLAY:-}"
if [ -n "$replay" ] && [ -n "$ctx_tokens" ]; then
    echo "CORRUPTION_REPLAY and CORRUPTION_CTX_TOKENS are two instruments - set one" >&2
    exit 2
fi
out="$root/decode_out/corruption-arms"
[ -n "$ctx_tokens" ] && out="$out-ctx$ctx_tokens-$position"
[ -n "$replay" ] && out="$out-replay-$replay"
[ -n "${CORRUPTION_TAG:-}" ] && out="$out-$CORRUPTION_TAG"
mkdir -p "$out"
port="${CORRUPTION_PORT:-8099}"
# The RAM threshold follows the pinned budget: with CROW_PINNED_BUDGET_GB set
# (a smaller cold tier, more experts hot in VRAM - numerics-neutral, #88 proved
# placement byte-identical), the ladder runs on a sessionized box whose balloon
# ceiling is below the full -M tier; every arm shares the same placement, so the
# overlay comparison stays exact. Pool recovery still needs the margin on top.
need_gib=$(( ${CROW_PINNED_BUDGET_GB:-44} + 2 ))

# #82 pool behavior: after every engine exit the ~46 GiB pinned tier sits in
# the NVIDIA driver pool; the next arm's loader sees MemAvailable ~8 GiB and
# refuses (manager.rs / residency.rs). Sustained anonymous pressure returns
# the pool. Same recovery gate-linux.sh and run-90-arm-final.sh run between
# every engine item (verified 2026-09-21, df5fe09) - imported verbatim,
# because without it this ladder dies after arm 1.
summary() {
python3 - "$out" <<'PY'
import json, sys, glob, os
rows = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.json"))):
    d = json.load(open(f))
    if "label" not in d:
        continue
    rows.append(d)
print("\n==== corruption arms summary ====")
print("%-10s %6s %6s %8s %10s %12s" % ("arm", "rounds", "lines", "badlines", "hexerr", "line_err_rate"))
for d in rows:
    print("%-10s %6d %6d %8d %10d %12s" % (
        d["label"], d["rounds_ok"], d["lines_total"], d["lines_with_error"],
        d["hex_char_errors"], d["line_error_rate"]))
PY
}

# HARD #82 state: two balloon passes in a row without a rise (9 -> 9 -> 9 GiB,
# 2026-09-21 21:45 and 22:47) never recovered in any later pass - only a
# reboot returns the pool. Say so, print the summary and exit instead of
# burning four more passes and hanging in wait_ram; finished arms are skipped
# on the next start.
pool_recover() {
    local av prev=-1 flat=0
    for i in 1 2 3 4 5 6; do
        av=$(awk '/^MemAvailable/{print int($2/1048576)}' /proc/meminfo)
        if [ "${av:-0}" -ge "$need_gib" ]; then
            [ "$i" -gt 1 ] && echo "  pool_recover: MemAvailable ${av} GiB after $((i-1)) pass(es)"
            return 0
        fi
        if [ "$prev" -ge 0 ] && [ "$av" -lt $((prev + 2)) ]; then flat=$((flat + 1)); else flat=0; fi
        if [ "$flat" -ge 2 ]; then
            echo "  pool_recover: HARD state - two passes without a rise (MemAvailable ${av} GiB, need $need_gib). REBOOT, then start the same command again."
            summary
            exit 3
        fi
        prev=$av
        echo "  pool_recover: pass $i (MemAvailable ${av} GiB, need $need_gib)"
        timeout 300 python3 "$root/decode_out/kv-ab/balloon.py" "$need_gib" >/dev/null 2>&1
    done
    av=$(awk '/^MemAvailable/{print int($2/1048576)}' /proc/meminfo)
    echo "  pool_recover: MemAvailable only ${av} GiB after 4 passes (need $need_gib) - arm may refuse"
}

wait_ram() {
    while true; do
        a=$(awk '/^MemAvailable/{print int($2/1048576)}' /proc/meminfo)
        [ "$a" -ge "$need_gib" ] && return 0
        sleep 10
    done
}

# After an arm: wait for serve's REAL exit, then for MemAvailable to stop
# rising on its own, and only then let pool_recover balloon. The first clean
# ladder (2026-09-21 20:58) ran baseline 8/8, then six balloon passes sat at
# MemAvailable 10 GiB and arm 2 never booted: run_arm slept 5 s after the kill,
# while serve frees the ~46 GiB pinned tier before exit (#82) - a balloon that
# presses against pages a draining serve still holds gets nothing back. Both
# waits are bounded and SAY what they saw, so the log settles which it was.
settle() {
    local t0=$SECONDS av prev=-1 flat=0
    while pgrep -f "release/serve .*--port $port" >/dev/null 2>&1; do
        if [ $((SECONDS - t0)) -ge 600 ]; then
            echo "  settle: serve STILL alive 600 s after SIGTERM - not escalating, the next arm will wait on RAM"
            break
        fi
        sleep 5
    done
    echo "  settle: serve exit after $((SECONDS - t0)) s"
    t0=$SECONDS
    while [ $((SECONDS - t0)) -lt 600 ]; do
        av=$(awk '/^MemAvailable/{print int($2/1048576)}' /proc/meminfo)
        [ "$av" -ge "$need_gib" ] && break
        if [ "$av" -le "$prev" ]; then flat=$((flat + 1)); else flat=0; fi
        [ "$flat" -ge 6 ] && break  # 90 s without a rise: the rest is the driver pool, balloon territory
        prev=$av
        sleep 15
    done
    echo "  settle: MemAvailable ${av} GiB after $((SECONDS - t0)) s of waiting (need $need_gib)"
}

run_arm() {  # label overlay_path_or_empty
    local label="$1" overlay="$2"
    pool_recover
    wait_ram
    echo "=== arm $label : $(date +%H:%M:%S) : MemAvailable $(awk '/^MemAvailable/{print int($2/1048576)}' /proc/meminfo)GiB"
    # CROW_CNQ_OVERLAY= EXPLICITLY EMPTY for the baseline: serve-linux.sh
    # defaults to the attn overlay since 723d18f, and a baseline that silently
    # carries it would compare overlay against overlay (2026-09-21, caught
    # before the first clean run).
    local envs=(env "CROW_RAM_MARGIN_GB=1" "CROW_CNQ_OVERLAY=")
    [ -n "$overlay" ] && envs+=("CROW_CNQ_OVERLAY=$overlay")
    "${envs[@]}" "$root/tools/serve-linux.sh" --port "$port" \
        >"$out/serve-$label.log" 2>&1 &
    local serve_pid=$!
    for _ in $(seq 1 120); do
        curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && break
        sleep 2
    done
    if ! curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
        echo "arm $label: serve did not come up (see $out/serve-$label.log)"
        kill "$serve_pid" 2>/dev/null
        wait "$serve_pid" 2>/dev/null
        settle
        return 1
    fi
    # plain python3, no systemd-run: the --pipe form dies from a nohup
    # terminal (stdin fd type), and the service form KILLED the serve scope
    # at its start (SIGTERM to serve 2 s after health OK, 2026-09-21 19:25,
    # 8x connection refused). The probe is a 100-line urllib script - it
    # needs no isolation and no memory cap.
    if [ -n "$replay" ]; then
        python3 "$root/tools/corruption-replay-probe.py" --port "$port" --label "$label" \
            --preset "$replay" ${CORRUPTION_REPLAY_SESSION:+--session "$CORRUPTION_REPLAY_SESSION"} \
            --sampling "${sampling:-{\}}" \
            --json "$out/$label.json" >"$out/probe-$label.log" 2>&1
    elif [ -n "$ctx_tokens" ]; then
        python3 "$root/tools/corruption-probe-long.py" --port "$port" --label "$label" \
            --ctx-tokens "$ctx_tokens" --position "$position" --sampling "${sampling:-{\}}" \
            --json "$out/$label.json" >"$out/probe-$label.log" 2>&1
    else
        python3 "$root/tools/corruption-probe.py" --port "$port" --label "$label" \
            --json "$out/$label.json" >"$out/probe-$label.log" 2>&1
    fi
    kill "$serve_pid" 2>/dev/null
    wait "$serve_pid" 2>/dev/null
    settle
}

# CORRUPTION_ARMS selects arms (default: all five). The pool goes into the HARD
# #82 state after one or two engine exits (2026-09-21 21:45: six balloon passes
# at 12 GiB after arm 2, ~48 GiB held with no owning process) and only a reboot
# returns it, so the ladder finishes across boots; the summary below reads every
# <label>.json in $out, whichever run wrote it.
overlay_for() {
    case "$1" in
        baseline)  echo "" ;;
        attn-ctrl) echo "$root/converter/layer91-attn-v-out-control.cnq" ;;
        attn-arm)  echo "$root/converter/layer91-attn-v-out-originals.cnq" ;;
        rule-arm)  echo "$root/converter/layer91-ffn-down-rule-originals.cnq" ;;
        all-arm)   echo "$root/converter/layer91-ffn-down-all-originals.cnq" ;;
        *) echo "unknown arm: $1" >&2; exit 2 ;;
    esac
}
for arm in ${CORRUPTION_ARMS:-baseline attn-ctrl attn-arm rule-arm all-arm}; do
    ov="$(overlay_for "$arm")" || exit 2
    # resume across boots: an arm with a complete 8/8 result is not rerun (each
    # engine exit spends the boot's pool budget; 2026-09-21 22:41 a restart
    # reran the finished attn-arm first). CORRUPTION_FORCE=1 reruns anyway.
    # The replay probe runs rounds x points and states completeness itself
    # ("complete"); a file that carries the key is judged by it alone, so a
    # partial replay at rounds_ok 8 of 24 is never taken for a finished arm.
    done_re='"rounds_ok": 8'
    grep -q '"complete":' "$out/$arm.json" 2>/dev/null && done_re='"complete": true'
    if [ -z "${CORRUPTION_FORCE:-}" ] && grep -q "$done_re" "$out/$arm.json" 2>/dev/null; then
        echo "=== arm $arm : already complete in $arm.json - skipped (CORRUPTION_FORCE=1 reruns)"
        continue
    fi
    run_arm "$arm" "$ov"
done

summary
