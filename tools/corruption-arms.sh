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
out="$root/decode_out/corruption-arms"
mkdir -p "$out"
port="${CORRUPTION_PORT:-8099}"
need_gib=45

wait_ram() {
    while true; do
        a=$(awk '/^MemAvailable/{print int($2/1048576)}' /proc/meminfo)
        [ "$a" -ge "$need_gib" ] && return 0
        sleep 10
    done
}

run_arm() {  # label overlay_path_or_empty
    local label="$1" overlay="$2"
    wait_ram
    echo "=== arm $label : $(date +%H:%M:%S) : available $(awk '/^MemAvailable/{print int($2/1048576)}' /proc/meminfo)GiB"
    local envs=(env "CROW_RAM_MARGIN_GB=1")
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
        return 1
    fi
    systemd-run --quiet --pipe --user -p MemoryMax=2G \
        python3 "$root/tools/corruption-probe.py" --port "$port" --label "$label" \
        --json "$out/$label.json"
    kill "$serve_pid" 2>/dev/null
    wait "$serve_pid" 2>/dev/null
    sleep 5
}

run_arm baseline ""
run_arm attn-ctrl "$root/converter/layer91-attn-v-out-control.cnq"
run_arm attn-arm "$root/converter/layer91-attn-v-out-originals.cnq"
run_arm rule-arm "$root/converter/layer91-ffn-down-rule-originals.cnq"
run_arm all-arm  "$root/converter/layer91-ffn-down-all-originals.cnq"

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
