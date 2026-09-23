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
# CORRUPTION_REPLAY_CROW_CORE pins the crow_core.py that builds the body (the
# probe's default is the INSTALLED one, and a reinstall changes the tools array:
# 2026-09-22 the session ran on crow 3dbc015, the install moved to 0e65d70 the
# same evening -- the byte-exact replay needs the old core, exported beside the
# snapshot). CORRUPTION_REPLAY_ROUNDS sets seeds per point (default 8); a
# greedy ladder (CORRUPTION_SAMPLING='{"temperature":0}') needs only 1.
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

# #103 (2026-09-23): the boot gate is the engine's own free_for_pin
# (tools/pin-room.sh -> engine ramcheck), not MemAvailable. The driver pool the
# old balloon pressed against is reclaimable and counted as free there; a
# refusal means live processes hold the RAM, and says which.
. "$root/tools/pin-room.sh"
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

pool_recover() {
    pin_room "$need_gib" && return 0
    echo "  pool_recover: free_for_pin below $need_gib GiB - live processes hold the RAM (see above); stopping, finished arms are skipped on the next start"
    summary
    exit 3
}

wait_ram() {
    while ! PIN_ROOM_WAIT_S=0 pin_room "$need_gib" >/dev/null; do
        sleep 10
    done
}

# After an arm: wait for serve's REAL exit (it frees the pinned tier before it
# goes, #82), bounded and said out loud; pool_recover then reads free_for_pin.
settle() {
    local t0=$SECONDS
    while pgrep -f "release/serve .*--port $port" >/dev/null 2>&1; do
        if [ $((SECONDS - t0)) -ge 600 ]; then
            echo "  settle: serve STILL alive 600 s after SIGTERM - not escalating, the next arm will wait on RAM"
            break
        fi
        sleep 5
    done
    echo "  settle: serve exit after $((SECONDS - t0)) s"
}

complete() {  # json_path -> 0 when the probe said it finished
    grep -q '"complete": true' "$1" 2>/dev/null
}

replay_probe() {  # label rounds_or_empty sampling_json
    local label="$1" rounds="$2" samp="$3"
    if complete "$out/$label.json"; then
        echo "  $label: already complete - skipped"
        return 0
    fi
    python3 "$root/tools/corruption-replay-probe.py" --port "$port" --label "$label" \
        --preset "$replay" ${CORRUPTION_REPLAY_SESSION:+--session "$CORRUPTION_REPLAY_SESSION"} \
        ${CORRUPTION_REPLAY_CROW_CORE:+--crow-core "$CORRUPTION_REPLAY_CROW_CORE"} \
        ${rounds:+--rounds "$rounds"} --sampling "$samp" \
        --json "$out/$label.json" >"$out/probe-$label.log" 2>&1
}

# THE TABLE IS WRITTEN AFTER EVERY ARM, not at the end: the ladder of record
# runs across several starts, and a result that only lives in a
# summary printed by the last boot is a result a reboot can lose. One row per
# probe file, appended once (the row carries the file's sha, a rerun of the
# table step never duplicates it). $out/TABLE.md is the durable record.
table_rows() {  # arm
python3 - "$out" "$1" <<'PY'
import hashlib, json, os, sys, time
out, arm = sys.argv[1], sys.argv[2]
table = os.path.join(out, "TABLE.md")
if not os.path.exists(table):
    with open(table, "w") as fh:
        fh.write("| written | arm | variant | complete | corrupt calls / calls | per point (K: kinds) | line_error_rate | file sha |\n")
        fh.write("|---|---|---|---|---|---|---|---|\n")
have = open(table).read()
for label, variant in ((arm, "seeded"), (arm + "-greedy", "greedy")):
    path = os.path.join(out, label + ".json")
    if not os.path.exists(path):
        continue
    raw = open(path, "rb").read()
    sha = hashlib.sha256(raw).hexdigest()[:12]
    if sha in have:
        continue
    d = json.loads(raw)
    pts = []
    for p in d.get("points") or []:
        bad = []
        for r in d.get("rounds_detail") or []:
            if r.get("at") != p.get("at"):
                continue
            kinds = sorted({e.get("kind") for c in r.get("calls") or []
                            for e in c.get("errors") or []})
            if r.get("error"):
                kinds = ["request error"]
            if kinds:
                bad.append("seed %s %s" % (r.get("seed"), "+".join(kinds)))
        pts.append("K=%s: %s" % (p.get("at"), ", ".join(bad) or "clean"))
    row = "| %s | %s | %s | %s | %s / %s | %s | %s | %s |\n" % (
        time.strftime("%Y-%m-%d %H:%M"), arm, variant, d.get("complete"),
        d.get("lines_with_error"), d.get("lines_total"), "; ".join(pts) or "-",
        d.get("line_error_rate"), sha)
    with open(table, "a") as fh:
        fh.write(row)
    print("  table: " + row.strip())
PY
}

run_arm() {  # label overlay_path_or_empty
    local label="$1" overlay="$2"
    pool_recover
    wait_ram
    echo "=== arm $label : $(date +%H:%M:%S) : MemAvailable $(awk '/^MemAvailable/{print int($2/1048576)}' /proc/meminfo)GiB"
    # CROW_CNQ_OVERLAY= EXPLICITLY EMPTY for the baseline: serve-linux.sh
    # defaulted to the attn overlay from 723d18f to 0efe3b5 and to the dense
    # overlay from 0924406 to 0254ed6 (2026-09-23; no overlay by default since),
    # and a baseline that silently carries one would compare overlay against
    # overlay (2026-09-21, caught before the first clean run). An empty value
    # attaches nothing (boot.rs), so the guard stays harmless.
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
        replay_probe "$label" "${CORRUPTION_REPLAY_ROUNDS:-}" "${sampling:-{\}}"
        # CORRUPTION_REPLAY_GREEDY=1: the same boot also answers under greedy
        # (one round -- greedy has one answer). A boot is the expensive unit:
        # 2026-09-22 every engine exit was followed by a MemAvailable gate that
        # asked for a reboot (#103: a wrong gate, fixed), so each
        # boot carries both questions -- seed 0 (the live condition) and
        # argmax (is the wrong digit the MOST likely token under this arm?).
        [ -n "${CORRUPTION_REPLAY_GREEDY:-}" ] && \
            replay_probe "$label-greedy" 1 '{"temperature":0}'
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

# CORRUPTION_ARMS selects arms (default: all five). Before the #103
# (2026-09-23) the balloon gate stalled after one or two engine exits
# (2026-09-21 21:45: six passes at 12 GiB) and the ladder was finished across
# reboots; the summary below still reads every <label>.json in $out, whichever
# run wrote it.
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
    if [ -n "$replay" ] && [ -n "${CORRUPTION_REPLAY_GREEDY:-}" ] \
        && ! complete "$out/$arm-greedy.json"; then
        done_re='never-matches-a-half-finished-arm'
    fi
    if [ -z "${CORRUPTION_FORCE:-}" ] && grep -q "$done_re" "$out/$arm.json" 2>/dev/null; then
        echo "=== arm $arm : already complete in $arm.json - skipped (CORRUPTION_FORCE=1 reruns)"
        continue
    fi
    run_arm "$arm" "$ov"
    [ -n "$replay" ] && table_rows "$arm"
done

summary
