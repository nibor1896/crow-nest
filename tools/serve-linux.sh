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

# pass the caller's CROW_* switches through the `env` that sets LD_LIBRARY_PATH
crow_env=()
while IFS= read -r kv; do crow_env+=("$kv"); done < <(env | grep '^CROW_' || true)

# #91 NO OVERLAY BY DEFAULT (2026-09-22, tools/results/91-corruption-ctx100000-end):
# the attn-v-out BF16 overlay was the default from 723d18f to here on the
# expectation that it removes the long-session corruption. Measured at 100k
# context it is the one arm that gets WORSE with depth: 28/320 dropped or
# wrong lines against the bare container's 8/320 (placebo 2/320), errors on
# three seeds where every other arm errs on one. The bare container of the
# parity record is the operating point again. An overlay is opt-in only:
# CROW_CNQ_OVERLAY=<file>.
if [ -n "${CROW_CNQ_OVERLAY:-}" ]; then
    echo "serve-linux.sh: CROW_CNQ_OVERLAY=$CROW_CNQ_OVERLAY (opt-in; the default is the bare container since the #91 100k ladder)" >&2
fi

printf 'serve-linux.sh: scope MemoryHigh=%s MemoryMax=%s MemorySwapMax=0, %s CROW_* passed through\n' \
    "$high" "$max" "${#crow_env[@]}" >&2

cd "$root"
exec systemd-run --user --scope --slice=session.slice --quiet \
    -p MemorySwapMax=0 -p "MemoryHigh=$high" -p "MemoryMax=$max" \
    env "LD_LIBRARY_PATH=$cuda_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        ${crow_env[@]+"${crow_env[@]}"} \
        "$bin" "$@"
