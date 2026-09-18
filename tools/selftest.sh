#!/usr/bin/env bash
# Usage, from anywhere:  tools/selftest.sh [package-dir] [--with-originals] [--full]
#
# The PACKAGE self-test (F5, issue #64, 2026-09-18). It answers one question for somebody who
# downloaded `nibor1896/Qwen3.8-Flash-Next-CNQ4.5-M` and has NO originals: does the quant this
# engine just loaded still compute what the unquantized model computes? It needs the package
# and the engine, and nothing else - no `models/`, no oracle venv, no torch, no network.
#
# Three items, GREEN/RED each, non-zero exit on any RED:
#
#   1. without the originals   the positive control: `test ! -d models` in the package directory.
#                              A tree that HAS the originals cannot prove anything about a
#                              download, so the run is REFUSED there - the engine is not started
#                              at all - unless --with-originals says out loud that this is the
#                              other arm.
#   2. golden sha256           the `selftest/` lines of the package's own SHA256SUMS, checked with
#                              `sha256sum -c`. The goldens ARE the reference: a golden that drifted
#                              would move the gate instead of failing it. --full checks every line
#                              of SHA256SUMS instead, which reads the 105 GB container (about 195 s
#                              at the 531 to 537 MB/s of SHA256SUMS.log, 2026-09-11).
#   3. decode selftest         the engine's layer outputs against the goldens, max_abs per layer
#                              against the manifest's gate (layer 0: 0.125, the layercheck gate of
#                              record since 2026-09-04). The mode's exit code is this item.
#
# Provenance of the numbers this script does NOT hard-code: the gate lives in
# `selftest/manifest.json` next to the goldens it gates, because the manifest travels with the
# package and this script does not. Values of record on the -M container (RTX 5090 / Arch Linux /
# driver 610.57.04 / CUDA 13.3.1, 2026-09-18): layer 0 max_abs 9.184837e-2 against the gate 0.125.
#
# Environment: CROW_CNQ / CROW_HOTSETS override the container and the hot-set manifest (the
# package's own files are the default). CUDA_LIB names the CUDA runtime directory (default
# ~/.local/share/crow/cuda/lib) and goes on LD_LIBRARY_PATH; DECODE_BIN names the `decode` binary
# (default <this repo>/engine/target/release/decode). NO_SCOPE=1 runs the engine outside the
# memory-bounded `systemd-run --user --scope` the rest of this repo uses.

set -u

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
pkg="$root"
with_originals=0
full=0
for a in "$@"; do
    case "$a" in
        --with-originals) with_originals=1 ;;
        --full) full=1 ;;
        -*) echo "selftest.sh: unknown option $a" >&2; exit 2 ;;
        *) pkg="$a" ;;
    esac
done
case "$pkg" in /*) ;; *) pkg="$(cd "$pkg" && pwd)" || exit 2 ;; esac

bin="${DECODE_BIN:-$root/engine/target/release/decode}"
cuda_lib="${CUDA_LIB:-$HOME/.local/share/crow/cuda/lib}"

red=0
green() { printf 'GREEN  %-28s %s\n' "$1" "${2:-}"; }
fail()  { printf 'RED    %-28s %s\n' "$1" "${2:-}"; red=1; }

[ -x "$bin" ] || { echo "selftest.sh: no $bin - build it with 'cd engine && cargo build --release', or set DECODE_BIN" >&2; exit 2; }
[ -d "$pkg/selftest" ] || { echo "selftest.sh: no $pkg/selftest - the golden set is part of the package" >&2; exit 2; }

# the package's own files first, the repo layout second (converter/ and decode_out/), CROW_* last word
cnq="${CROW_CNQ:-}"
if [ -z "$cnq" ]; then
    for c in "$pkg/Qwen3.8-Flash-Next-CNQ4.5-M.cnq" "$pkg/converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq"; do
        [ -f "$c" ] && { cnq="$c"; break; }
    done
fi
hot="${CROW_HOTSETS:-}"
if [ -z "$hot" ]; then
    for h in "$pkg/hotsets-M-longctx2100-n160.json" "$pkg/decode_out/hotsets-M-longctx2100-n160.json"; do
        [ -f "$h" ] && { hot="$h"; break; }
    done
fi
[ -n "$cnq" ] && [ -f "$cnq" ] || { echo "selftest.sh: no container in $pkg - set CROW_CNQ" >&2; exit 2; }
[ -n "$hot" ] && [ -f "$hot" ] || { echo "selftest.sh: no hot-set manifest in $pkg - set CROW_HOTSETS" >&2; exit 2; }

echo "== selftest.sh  $(date -Is)  package $pkg"
echo "   container $cnq  ($(stat -c%s "$cnq") B)"
echo "   hot sets  $hot"
echo "   decode    $bin"

# ---- 1) the positive control -------------------------------------------------------------------
# `test ! -d models` is the whole control, and it is deliberately a DIRECTORY test and not a test
# for the safetensors: a self-test whose verdict depends on a file that may or may not be inside
# models/ is not a control. A tree with models/ is the other arm and says so with the flag.
control="test ! -d models"
if [ ! -d "$pkg/models" ]; then
    green "without the originals" "$control holds in the package directory"
elif [ "$with_originals" = 1 ]; then
    green "WITH the originals" "--with-originals: $pkg/models exists, so this run proves the numbers, NOT the independence"
else
    fail "without the originals" "$pkg/models exists - pass --with-originals to run the other arm on purpose"
    # a REFUSAL, not a warning: the engine is not started at all, because a run in a tree that
    # holds the originals cannot be presented as a run without them whatever its numbers say
    echo "== selftest.sh: RED - refused before the engine was started"
    exit 1
fi
for extra in oracle/golden .venv-oracle; do
    [ -e "$pkg/$extra" ] && echo "   note: $pkg/$extra is present; the self-test reads neither (it reads $pkg/selftest only)"
done

# ---- 2) the golden set's own sha256 ------------------------------------------------------------
if [ ! -f "$pkg/SHA256SUMS" ]; then
    fail "golden sha256" "no $pkg/SHA256SUMS"
elif [ "$full" = 1 ]; then
    if ( cd "$pkg" && sha256sum -c SHA256SUMS ); then green "package sha256 (--full)" "every line of SHA256SUMS OK"
    else fail "package sha256 (--full)" "sha256sum -c SHA256SUMS failed"; fi
else
    lines=$(grep -c ' selftest/' "$pkg/SHA256SUMS")
    if [ "${lines:-0}" -lt 1 ]; then
        fail "golden sha256" "SHA256SUMS carries no selftest/ line"
    elif ( cd "$pkg" && grep ' selftest/' SHA256SUMS | sha256sum -c - ); then
        green "golden sha256" "$lines of $lines selftest/ lines OK (--full also reads the container)"
    else
        fail "golden sha256" "a selftest/ golden does not match SHA256SUMS"
    fi
fi

# ---- 3) the engine against the goldens ---------------------------------------------------------
# one engine at a time, and never on a busy GPU - the same rule the parity gate runs under
if command -v nvidia-smi > /dev/null 2>&1; then
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "${used:-0}" -ge 2000 ] && { fail "decode selftest" "GPU holds ${used} MiB - refusing to run"; echo "== selftest.sh: RED"; exit 1; }
fi
if pgrep -a -x 'serve|decode|parity' > /dev/null 2>&1; then
    fail "decode selftest" "an engine is alive - refusing to run"
    echo "== selftest.sh: RED"
    exit 1
fi

runner=(env)
if [ "${NO_SCOPE:-0}" != 1 ] && command -v systemd-run > /dev/null 2>&1; then
    runner=(systemd-run --user --scope --slice=session.slice --quiet
            -p MemorySwapMax=0 -p MemoryHigh=52G -p MemoryMax=54G env)
fi
log="${SELFTEST_LOG:-$pkg/selftest/selftest.log}"
t0=$(date +%s%3N)
# cwd is the PACKAGE, the way a downloader would run it; every path handed to the engine is
# absolute anyway, and the lock stays the engine crate's own `.engine.lock`
( cd "$pkg" && "${runner[@]}" CROW_CNQ="$cnq" CROW_HOTSETS="$hot" CROW_GRAPH=1 CROW_MMA=1 \
    "LD_LIBRARY_PATH=$cuda_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "$bin" selftest "$pkg/selftest" ) > "$log" 2>&1
rc=$?
t1=$(date +%s%3N)
wall=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.1f", (b-a)/1000}')
grep -E '^selftest: (layer|PASS|FAIL)' "$log"
if [ $rc -eq 0 ]; then green "decode selftest" "exit 0  ${wall}s  log $log"
else fail "decode selftest" "exit $rc  ${wall}s  see $log"; fi

echo "== selftest.sh: $([ $red -eq 0 ] && echo 'ALL GREEN' || echo 'RED - see above')"
exit $red
