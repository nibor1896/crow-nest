#!/bin/bash
# #214 acceptance, 2026-10-11 (robin's GPU go): glm5_run arms on the quick decode (57 ids, -n 128, --cold), arm3.env.
#   A = --kv-rows context, CROW_GLM_KV_LAZY unset   (serve's KV and plan, eager)
#   B = --kv-rows context, CROW_GLM_KV_LAZY=4096    (serve's KV and plan, lazy)
#   C = no --kv-rows, switch unset                  (cap = prompt + n, the 2,524-slot reference)
# Alternated A B C x 3. The caller holds C:\Users\robin\dev\.gpu-lock. Usage: gr.sh [reps]
R=/c/Users/robin/dev/crow-nest; D=$R/runs/glm53-flash/kvlazy-20261011; E=$R/engine; K=${1:-3}
export CROW_CNQ='C:\Users\robin\dev\crow-nest\converter\GLM-5.3-Flash-MUL1K3.cnq'
TOK='C:\Users\robin\dev\crow-nest\models\GLM-5.3-Flash-original\tokenizer.json'; export CROW_TOKENIZER="$TOK"
ARM3="$(cat $R/runs/glm53-flash/quick/arm3.env)"
TXT="Explain in detail how a modern CPU cache hierarchy works: L1, L2, L3, coherence protocols, write-back versus write-through, and how false sharing hurts multithreaded code. Give concrete examples."
cd $E || exit 1
echo "gr start $(date +%T) head $(git rev-parse --short HEAD) dirty $(git status --porcelain -- . | wc -l) bin $(date -r target/release/glm5_run.exe '+%F %T')" | tee -a $D/gr.log
run() { # label rep
  local L=$1 i=$2 X=() F=()
  case $L in A) F=(--kv-rows context);; B) F=(--kv-rows context); X=(CROW_GLM_KV_LAZY=4096);; C) ;; esac
  echo "== gr-$L-r$i $(date +%T) vram_used_mib $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) extra: ${X[*]} ${F[*]}" | tee -a $D/gr.log
  env -u CROW_GLM_KV_LAZY -u CROW_GLM_MTP $ARM3 "${X[@]}" timeout 400 target/release/glm5_run --cold --reps 1 --tokenizer "$TOK" \
     --prompt "$TXT" -n 128 "${F[@]}" --json $D/gr-$L-r$i.json 2>&1 | sed -E 's/  \[l[0-9]+ [0-9/ l]*\]$//' > $D/gr-$L-r$i.log
  echo "rc=${PIPESTATUS[0]} $(date +%T)" | tee -a $D/gr.log
  grep -hE "rep 1/1 .*decode:|KV after rep|refused" $D/gr-$L-r$i.log | tee -a $D/gr.log
}
for i in $(seq 1 $K); do for L in A B C; do run $L $i; done; done
echo "gr done $(date +%T)" | tee -a $D/gr.log
