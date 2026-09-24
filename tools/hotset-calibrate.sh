#!/usr/bin/env bash
# Hot-set calibration on real Crow traffic (2026-09-24, after the PLE fix 85a48e7).
# One `decode routestats <ids> <out> 0` per corpus file: a prefill, then the
# per-expert selection counts of every routed choice (hot or cold) as JSON.
# tools/hotset-from-counts.py sums them and writes the sidecar; a held-out
# file's counts score any sidecar offline. Same scope and env as gate-linux.sh.
# With ROUTE_DUMP=1 each run also writes <out>/<name>-routes.bin
# (CROW_ROUTE_DUMP_PREFILL: every prefill position's routed ids, per chunk and layer).
# Usage, from the repository root:
#   [ROUTE_DUMP=1] tools/hotset-calibrate.sh <out-dir> <ids.json>...
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
bin="$root/engine/target/release/decode"
cuda_lib="${CUDA_LIB:-$HOME/.local/share/crow/cuda/lib}"
out="$1"; shift
mkdir -p "$out"

procs=$(pgrep -a -x 'serve|decode|parity|llama.*' | tr '\n' ' ' || true)
if [ -n "$procs" ]; then echo "hotset-calibrate: an engine is alive ($procs) - refusing" >&2; exit 1; fi
. "$root/tools/pin-room.sh"

for ids in "$@"; do
    name=$(basename "$ids" -ids.json)
    pin_room 46 || echo "  free_for_pin below 46 GiB - the run may refuse"
    echo "$(date -Is) routestats $name ($(python3 -c "import json,sys;print(len(json.load(open(sys.argv[1]))))" "$ids") tokens)"
    dump=()
    if [ "${ROUTE_DUMP:-0}" = 1 ]; then dump=("CROW_ROUTE_DUMP_PREFILL=$out/$name-routes.bin"); fi
    systemd-run --user --scope --slice=session.slice --quiet \
        -p MemorySwapMax=0 -p MemoryHigh=52G -p MemoryMax=54G \
        env CROW_CNQ=converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq \
            CROW_HOTSETS=decode_out/hotsets-M-longctx2100-n160.json \
            CROW_MMA=1 ${dump[@]+"${dump[@]}"} "LD_LIBRARY_PATH=$cuda_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
            "$bin" routestats "$ids" "$out/$name-counts.json" 0 > "$out/$name.log" 2>&1
    echo "$(date -Is) done: $(grep -E 'routestats: prefill' "$out/$name.log" || tail -1 "$out/$name.log")"
done
