#!/usr/bin/env bash
# Start `serve` on Linux inside a memory-bounded transient scope (issue #15).
#
# Why a scope: the cold tier is pinned, unevictable and unswappable, so the
# kernel's only reclaim target left is the page cache. systemd-oomd acts on
# memory PRESSURE, not on exhaustion, and a machine-wide pressure spike gets
# the desktop killed alongside the engine. A scope with MemoryHigh/MemoryMax
# turns that into a bounded cgroup: the engine is throttled, then killed alone.
# MemorySwapMax=0 keeps zram out of it - swapping a pinned tier is not possible
# and swapping the rest only adds latency. Same shape Crow uses for
# llama-server on Linux (Crow commits bfcec1f, acc2742).
#
# Limits are computed from /proc/meminfo MemTotal: MemoryHigh = MemTotal - 8G,
# MemoryMax = MemTotal - 6G, so the desktop keeps a floor the engine cannot eat.
#
# Since the #103 (2026-09-23) the cold tier is registered ANONYMOUS
# memory (CROW_PINNED_ALLOC=register), so it is charged to this scope: a 45.1 GiB
# tier boots at memory.current ~49.9 GiB (tier + heap + page cache), under
# MemoryHigh 54.2 GiB on the 62 GiB host. The old write-combined tier
# (CROW_PINNED_ALLOC=wc) was driver memory and escaped the scope entirely
# (memory.current 3.0 GiB with the same tier pinned). A pinned budget near
# 50 GiB therefore runs close to MemoryHigh; the page cache is what gets reclaimed
# first, the pinned tier cannot be.
#
# Environment: CUDA_LIB names the CUDA runtime directory (default
# ~/.local/share/crow/cuda/lib; never the .../lib/stubs sibling). Every CROW_*
# variable of the caller is passed through. Every argument is passed to `serve`.
#
# Usage, from anywhere:   tools/serve-linux.sh --port 8099 [serve args...]
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bin="$root/engine/target/release/serve"
if [ ! -x "$bin" ]; then
    echo "serve-linux.sh: $bin is not there - build it with 'cd engine && cargo build --release --bin serve'" >&2
    exit 2
fi

cuda_lib="${CUDA_LIB:-$HOME/.local/share/crow/cuda/lib}"
if [ ! -d "$cuda_lib" ]; then
    echo "serve-linux.sh: no CUDA runtime directory at $cuda_lib - set CUDA_LIB" >&2
    exit 2
fi

mem_total=$(( $(awk '/^MemTotal:/ {print $2}' /proc/meminfo) * 1024 ))
high=$(( mem_total - 8 * 1024 * 1024 * 1024 ))
max=$(( mem_total - 6 * 1024 * 1024 * 1024 ))
if [ "$high" -le 0 ] || [ "$max" -le 0 ]; then
    echo "serve-linux.sh: MemTotal $mem_total B is too small to leave the 8G/6G floor" >&2
    exit 2
fi

# #91 (2026-09-23): NO OVERLAY BY DEFAULT again. The corruption's cause was the PLE
# row read at the wrong container offset (85a48e7); the dense BF16 overlay only masked
# part of it and cost the operating point: pinned 50 GiB write-combined (the serve was
# OOM-killed at 18:38 CEST with the desktop left ~14 GiB) and a hot set of 128 instead
# of 156 (prefill 500-600 tok/s, decode 33-37 tok/s). An overlay stays opt-in:
# CROW_CNQ_OVERLAY=<file> (CROW_CNQ_OVERLAY=none is accepted and means none).
if [ "${CROW_CNQ_OVERLAY:-}" = "none" ]; then
    unset CROW_CNQ_OVERLAY
fi

# pass the caller's CROW_* switches through the `env` that sets LD_LIBRARY_PATH -
# except secrets: a CROW_*_KEY / _TOKEN / _SECRET belongs to Crow the client
# (CROW_TAVILY_KEY, crow_core.py), the engine has no read site for it
# (docs/env.md), and it has no business in serve's process environment.
crow_env=()
while IFS= read -r kv; do crow_env+=("$kv"); done \
    < <(env | grep '^CROW_' | grep -vE '^CROW_[A-Z0-9_]*(KEY|TOKEN|SECRET)=' || true)

# The attn-v-out overlay (default 723d18f..2026-09-22) got WORSE at 100k context
# (tools/results/91-corruption-ctx100000-end: 28/320 wrong lines vs bare 8/320).
# The dense overlay above is a different, measured set; see the #91 block.
echo "serve-linux.sh: CROW_CNQ_OVERLAY=${CROW_CNQ_OVERLAY:-none (bare container)} CROW_PINNED_BUDGET_GB=${CROW_PINNED_BUDGET_GB:-unset} CROW_PINNED_ALLOC=${CROW_PINNED_ALLOC:-default}" >&2

# #102: CROW_KV=bf16 (the KV cache in bf16 instead of FP8 E4M3) is read by serve and
# passed through like every CROW_* above. It doubles the KV bytes, the
# planner pays with hot experts, and the larger cold tier no longer fits the 46 GiB
# default cap: set CROW_PINNED_BUDGET_GB with it (docs/env.md, CROW_KV row).
if [ -n "${CROW_KV:-}" ]; then
    echo "serve-linux.sh: CROW_KV=$CROW_KV (KV cache dtype; bf16 needs a larger CROW_PINNED_BUDGET_GB, now ${CROW_PINNED_BUDGET_GB:-unset})" >&2
fi

printf 'serve-linux.sh: scope MemoryHigh=%s MemoryMax=%s MemorySwapMax=0, %s CROW_* passed through\n' \
    "$high" "$max" "${#crow_env[@]}" >&2

cd "$root"
exec systemd-run --user --scope --slice=session.slice --quiet \
    -p MemorySwapMax=0 -p "MemoryHigh=$high" -p "MemoryMax=$max" \
    env "LD_LIBRARY_PATH=$cuda_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        ${crow_env[@]+"${crow_env[@]}"} \
        "$bin" "$@"
