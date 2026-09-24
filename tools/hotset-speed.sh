#!/usr/bin/env bash
# Criterion C of decode_out/hotset-0924/PREREG.md: decode tok/s of two sidecars on the
# same prompt, `decode run <ids> 64`, 3 runs each, alternating (old, new, old, ...) so
# drift hits both. One engine at a time, same scope and env as gate-linux.sh.
# Usage, from the repository root: tools/hotset-speed.sh <out-dir> <ids.json> <sidecar-a> <sidecar-b>
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
bin="$root/engine/target/release/decode"
cuda_lib="${CUDA_LIB:-$HOME/.local/share/crow/cuda/lib}"
out="$1"; ids="$2"; a="$3"; b="$4"
mkdir -p "$out"
. "$root/tools/pin-room.sh"
for rep in 1 2 3; do
    for sc in "$a" "$b"; do
        name="$(basename "$sc" .json)-r$rep"
        procs=$(pgrep -a -x 'serve|decode|parity|llama.*' | tr '\n' ' ' || true)
        if [ -n "$procs" ]; then echo "an engine is alive ($procs) - refusing" >&2; exit 1; fi
        pin_room 46 || true
        systemd-run --user --scope --slice=session.slice --quiet \
            -p MemorySwapMax=0 -p MemoryHigh=52G -p MemoryMax=54G \
            env CROW_CNQ=converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq CROW_HOTSETS="$sc" \
                CROW_GRAPH=1 CROW_MMA=1 "LD_LIBRARY_PATH=$cuda_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
                "$bin" run "$ids" 64 > "$out/$name.log" 2>&1
        echo "$(date -Is) $name: $(grep -E '^decode: mean' "$out/$name.log" || tail -1 "$out/$name.log")"
    done
done
