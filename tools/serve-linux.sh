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

# #91 DEFAULT OPERATING POINT (2026-09-21): the attn-v-out BF16 overlay --
# v_proj/o_proj/linear_attn.out_proj of every layer at full precision, the
# exact tensor rule llama.cpp holds in its highest tier short of f16 (the
# engine that won the blind comparison). The corruption class of the long
# sessions (trailing digits, df5fe00 for df5fe09) is the quantization
# residual this removes. Caller wins: set CROW_CNQ_OVERLAY yourself and that
# file is used; set it EMPTY (CROW_CNQ_OVERLAY=) for the bare no-overlay
# engine of the parity record.
if [ -z "${CROW_CNQ_OVERLAY+x}" ]; then
    overlay="$root/converter/layer91-attn-v-out-originals.cnq"
    if [ -f "$overlay" ]; then
        export CROW_CNQ_OVERLAY="$overlay"
        echo "serve-linux.sh: CROW_CNQ_OVERLAY default -> attn-v-out bf16 overlay (set CROW_CNQ_OVERLAY= to disable)" >&2
    else
        echo "serve-linux.sh: default overlay missing ($overlay) - serving the bare container" >&2
    fi
fi

printf 'serve-linux.sh: scope MemoryHigh=%s MemoryMax=%s MemorySwapMax=0, %s CROW_* passed through\n' \
    "$high" "$max" "${#crow_env[@]}" >&2

cd "$root"
exec systemd-run --user --scope --slice=session.slice --quiet \
    -p MemorySwapMax=0 -p "MemoryHigh=$high" -p "MemoryMax=$max" \
    env "LD_LIBRARY_PATH=$cuda_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        ${crow_env[@]+"${crow_env[@]}"} \
        "$bin" "$@"
