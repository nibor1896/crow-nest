#!/usr/bin/env bash
# The Linux twin of quick.sh (#230, 2026-10-14): the same two runs (one cold 8192-token prefill of
# prefill8192.ids, 128 cold decode tokens on the same question), with this checkout's own paths and each
# glm5_run inside the bounded scope tools/serve-linux.sh and tools/gate-linux.sh use (MemorySwapMax=0,
# MemoryHigh=52G, MemoryMax=54G in session.slice; the OOM note of #15). quick.sh stays the Windows script.
# Usage, from anywhere:  runs/glm53-flash/quick/quick-linux.sh <label> [extra env...]
#   operating set:       quick-linux.sh <label> $(cat runs/glm53-flash/quick/arm4.env) \
#                          CROW_GLM_ARENA_WARM=../runs/glm53-flash/arena-warm/glm53-cal4-decode-counts.json
# -> runs/glm53-flash/quick/<label>/  (glm5_run runs from engine/, so relative paths in the env are from there)
set -u
Q=$(cd "$(dirname "$0")" && pwd); R=$(cd "$Q/../../.." && pwd); L=$1; shift; D=$Q/$L; mkdir -p "$D"
E=${ENGINE_DIR:-$R/engine}
cuda_lib="${CUDA_LIB:-$HOME/.local/share/crow/cuda/lib}"
export CROW_CNQ="$R/converter/GLM-5.3-Flash-MUL1K3.cnq"
TOK="$R/models/GLM-5.3-Flash-original/tokenizer.json"; export CROW_TOKENIZER="$TOK"
# as quick.sh: arm2.env + CROW_GLM_RT2=1 under the caller's extra env (a later assignment wins in env)
ARM="$(cat "$Q/arm2.env") CROW_GLM_RT2=1"
run() {
    systemd-run --user --scope --slice=session.slice --quiet -p MemorySwapMax=0 -p MemoryHigh=52G -p MemoryMax=54G \
        env "LD_LIBRARY_PATH=$cuda_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" $ARM "$@"
}
cd "$E"
echo "start $(date +%T) head $(git rev-parse --short HEAD) builds $(pgrep -c -x 'cargo|rustc') extra: $*" | tee "$D/run.log"
run "$@" timeout 150 target/release/glm5_run --cold --reps 1 --prompt-ids "$Q/prefill8192.ids" -n 4 --json "$D/prefill.json" > "$D/prefill.log" 2>&1; echo "prefill rc=$?" | tee -a "$D/run.log"
run "$@" timeout 120 target/release/glm5_run --cold --reps 1 --tokenizer "$TOK" --prompt "Explain in detail how a modern CPU cache hierarchy works: L1, L2, L3, coherence protocols, write-back versus write-through, and how false sharing hurts multithreaded code. Give concrete examples." -n 128 --json "$D/decode.json" > "$D/decode.log" 2>&1; echo "decode rc=$?" | tee -a "$D/run.log"
grep -hE "rep 1/1 .*(prefill:|decode:)" "$D/prefill.log" "$D/decode.log" | sed -E 's/.*(prefill|decode): /\1: /' | tee -a "$D/run.log"
echo "done $(date +%T)" | tee -a "$D/run.log"
