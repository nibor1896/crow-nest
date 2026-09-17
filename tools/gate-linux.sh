#!/usr/bin/env bash
# Usage, from the repo root:  tools/gate-linux.sh [outdir]   (default decode_out/gate) - runs the five
# cheap gates of record (parity 8 / 512 / P8 teacher-forced, decode run 32, host-side checks), prints
# GREEN/RED per item and exits non-zero on any RED.
#
# Why this script exists: every commit on this branch has to reproduce the SAME bytes, and the values
# below are the values of record, not tuning knobs. Nothing here may be edited to make a gate pass.
#
# Provenance of every expected value (all measured on RTX 5090 / Arch Linux / driver 610.57.04 /
# CUDA 13.3.1 / NVRTC 13.3.33, inside the memory-bounded scope this script sets up):
#
#   8 rows    bceba6ff7724...  commit 9f12429 "Linux port": byte-identical to the WINDOWS reference
#                              decode_out/parity-62d-ref8 - the one form that survived the toolchain
#                              drift. 11,919,360 B = 12 rows x 248,320 f32. Re-confirmed at 0c9feb5,
#                              74c79f2, bb9d2ca, 7ddd296, 0667e0b.
#   512 rows  838723470927...  commit 9f12429: the LINUX value of record. The Windows bytes differ
#                              (logits drift from row 23, ids identical in all 517 positions) - that
#                              is the NVRTC/driver JIT, not the port (9f12429 proved it over four
#                              configurations). Re-confirmed at 0c9feb5, 74c79f2, bb9d2ca, 7ddd296.
#   P8 tf     3bb3e69edf90...  commit 9f12429, decode_out/linux-port/parity-p8tf: the teacher-forced
#                              form (prefill 8 ids, the other 504 fed through decode_step) - it puts
#                              the DECODE path under the parity contract. Re-confirmed at 74c79f2,
#                              bb9d2ca, 7ddd296.
#   run 32    the 32 ids       commit bb9d2ca ("the 32 generated ids identical to the 74c79f2
#                              binary") and 7ddd296 ("decode run 32 ids identical").
#   tests 153 / clippy 1422    commit 7ddd296 and 0667e0b gave 144 = 84 lib + 60 serve; TASK H
#                              (2026-09-17) added the three `cnq::tests::page_runs_*` unit tests of
#                              the PLE row fetch, so 147 = 87 lib + 60 serve. TASK J (2026-09-17)
#                              added six tests of the tool-call `arguments` contract - three in
#                              `toolcall.rs` for the invariant that an abandoned call still leaves a
#                              parseable JSON object, three in `bin/serve.rs` for the normalization,
#                              the rewrite note and the neighbouring template hazards - so
#                              153 = 90 lib + 63 serve. Clippy is unchanged at
#                              1422: it is the --all-targets form counted as grep -cE '^warning: ',
#                              the form the 1494 -> 1480 -> 1426 -> 1422 series was counted with.
#
# Environment: CROW_CNQ / CROW_HOTSETS / CROW_GRAPH / CROW_MMA are set here exactly as the runs of
# record had them; CUDA_LIB names the CUDA runtime directory (default ~/.local/share/crow/cuda/lib).
# Every engine run goes inside `systemd-run --user --scope` with MemorySwapMax=0 / MemoryHigh=52G /
# MemoryMax=54G - the same bounded cgroup tools/serve-linux.sh uses (issue #15). Runs are SEQUENTIAL
# on purpose: the RAM gate refuses a second engine while the first one holds the pinned tier.

set -u

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out="${1:-$root/decode_out/gate}"
case "$out" in /*) ;; *) out="$root/$out" ;; esac
bin="$root/engine/target/release/decode"
cuda_lib="${CUDA_LIB:-$HOME/.local/share/crow/cuda/lib}"

# ---- the values of record (see the provenance block above) --------------------------------------
SHA8="bceba6ff772431dedf57631e83c7418d3f0bca3ab7e33da828f8da5d12a122a2"
BYTES8="11919360"
SHA512="8387234709271515b091b1c4dbd0d59c66550d0e3feab551a6418d30b55c9105"
SHAP8="3bb3e69edf90a6c3839222d1ceae7fe06aed1ba49813daa1f7487e3c6e7cff2d"
IDS32="[13, 248046, 198, 248045, 74455, 198, 248068, 198, 760, 1156, 682, 3766, 264, 11316, 25, 328, 760, 3841, 13477, 37550, 33075, 888, 279, 15217, 5388, 1149, 271, 1919, 7701, 310, 381, 264]"
TESTS="153"
CLIPPY="1422"

red=0
green() { printf 'GREEN  %-28s %s\n' "$1" "${2:-}"; }
fail()  { printf 'RED    %-28s %s\n' "$1" "${2:-}"; red=1; }

[ -x "$bin" ] || { echo "gate-linux.sh: no $bin - build it with 'cd engine && cargo build --release'" >&2; exit 2; }
[ -d "$cuda_lib" ] || { echo "gate-linux.sh: no CUDA runtime directory at $cuda_lib - set CUDA_LIB" >&2; exit 2; }
mkdir -p "$out"
cd "$root"

# one engine at a time, and never on a busy GPU: a second pinned tier is exactly what the RAM gate
# is there to refuse, and a busy GPU would make the run fail for a reason that is not the code.
precheck() {
    local used procs
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    # -x: match the process NAME exactly. Without it the pattern matched the
    # comm of an unrelated `tmux: server` and refused every engine item
    # (2026-09-17, TASK H) - the gate then printed RED for a machine that was idle.
    procs=$(pgrep -a -x 'serve|decode|parity|llama.*' | tr '\n' ' ')
    if [ "${used:-9999}" -ge 2000 ]; then echo "  precheck: GPU holds ${used} MiB - refusing to run" >&2; return 1; fi
    if [ -n "$procs" ]; then echo "  precheck: an engine is alive ($procs) - refusing to run" >&2; return 1; fi
    return 0
}

# scope_run <logfile> <extra env assignments...> -- <args to decode>
scope_run() {
    local log="$1"; shift
    local extra=()
    while [ "$1" != "--" ]; do extra+=("$1"); shift; done
    shift
    systemd-run --user --scope --slice=session.slice --quiet \
        -p MemorySwapMax=0 -p MemoryHigh=52G -p MemoryMax=54G \
        env CROW_CNQ=converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq \
            CROW_HOTSETS=decode_out/hotsets-M-longctx2100-n160.json \
            CROW_GRAPH=1 CROW_MMA=1 "LD_LIBRARY_PATH=$cuda_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
            ${extra[@]+"${extra[@]}"} \
            "$bin" "$@" > "$log" 2>&1
}

# parity_item <name> <ids.json> <expected sha> <expected bytes|-> <extra env...>
parity_item() {
    local name="$1" ids="$2" want="$3" wantb="$4"; shift 4
    local dir="$out/$name" log="$out/$name.log"
    precheck || { fail "$name" "precheck refused"; return; }
    rm -rf "$dir"
    local t0 t1
    t0=$(date +%s%3N)
    scope_run "$log" "$@" -- parity "$ids" "$dir"
    local rc=$?
    t1=$(date +%s%3N)
    local wall; wall=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.1f", (b-a)/1000}')
    if [ $rc -ne 0 ]; then fail "$name" "decode exit $rc, see $log"; return; fi
    local got bytes
    got=$(sha256sum "$dir/gpu-logits.f32" | cut -d' ' -f1)
    bytes=$(stat -c%s "$dir/gpu-logits.f32")
    if [ "$got" != "$want" ]; then fail "$name" "sha256 $got != $want (${wall}s)"; return; fi
    if [ "$wantb" != "-" ] && [ "$bytes" != "$wantb" ]; then fail "$name" "$bytes B != $wantb B (${wall}s)"; return; fi
    green "$name" "sha256 ${got:0:12} $bytes B  ${wall}s"
}

echo "== gate-linux.sh  $(date -Is)  $(git -C "$root" rev-parse --short HEAD 2>/dev/null)  -> $out"

# 1) parity 8 rows - the form that is byte-identical to Windows
parity_item parity8 decode_out/parity-ids.json "$SHA8" "$BYTES8"

# 2) parity 512 rows - the Linux value of record
parity_item parity512 decode_out/real512-ids.json "$SHA512" -

# 3) P8 teacher-forced - the decode path under the parity contract (graph off by construction)
parity_item p8tf decode_out/real512-ids.json "$SHAP8" - CROW_GRAPH=0 CROW_PARITY_PREFILL=8

# 6) decode run 32 - the 32 generated ids of record
if precheck; then
    log="$out/run32.log"
    t0=$(date +%s%3N)
    scope_run "$log" -- run decode_out/parity-ids.json 32 "$out/run32"
    rc=$?
    t1=$(date +%s%3N)
    wall=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.1f", (b-a)/1000}')
    cp -f "$root/decode_out/run.json" "$out/run32.json" 2>/dev/null
    if [ $rc -ne 0 ]; then
        fail run32 "decode exit $rc, see $log"
    else
        gotids=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['trace'])" "$out/run32.json")
        if [ "$gotids" = "$IDS32" ]; then green run32 "32 ids of record  ${wall}s"; else fail run32 "ids differ: $gotids"; fi
    fi
else
    fail run32 "precheck refused"
fi

# 10) host-side checks - no GPU, no model
( cd "$root/engine" && cargo test --release > "$out/cargo-test.log" 2>&1 )
tpass=$(grep -hoE '^test result: ok\. [0-9]+ passed' "$out/cargo-test.log" | awk '{s+=$4} END{print s+0}')
tfail=$(grep -hoE '[0-9]+ failed' "$out/cargo-test.log" | awk '{s+=$1} END{print s+0}')
if [ "$tpass" = "$TESTS" ] && [ "$tfail" = "0" ]; then green "cargo test" "$tpass passed, 0 failed"
else fail "cargo test" "$tpass passed / $tfail failed, expected $TESTS / 0"; fi

( cd "$root/engine" && cargo clippy --release --all-targets > "$out/clippy.log" 2>&1 )
cw=$(grep -cE '^warning: ' "$out/clippy.log")
if [ "$cw" = "$CLIPPY" ]; then green clippy "$cw warnings"; else fail clippy "$cw warnings, expected $CLIPPY"; fi

for t in check_env_docs check_readme_dates check_model_card_dates; do
    if [ -f "$root/tools/$t.py" ]; then
        if PYTHONIOENCODING=utf-8 python3 "$root/tools/$t.py" > "$out/$t.log" 2>&1
        then green "$t" "exit 0"; else fail "$t" "exit $?, see $out/$t.log"; fi
    else
        green "$t" "not in this tree - skipped"
    fi
done

echo "== gate-linux.sh: $([ $red -eq 0 ] && echo 'ALL GREEN' || echo 'RED - see above')"
exit $red
