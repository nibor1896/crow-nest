#!/bin/bash
# run.sh's arms A B C without its own lock/rebuild (the caller holds C:\Users\robin\dev\.gpu-lock and built 045bfa0).
# Usage: job2.sh N [extra env applied to all arms, e.g. CROW_GLM_KV_LAZY=4096]
W=/c/Users/robin/dev; R=$W/crow-nest; D=$R/runs/glm53-flash/mtp-long-20261011; E=$R/engine; N=${1:-5500}; shift
ALL=("$@")
cd $E || exit 1
echo "job2 start $(date +%T) binary $(date -r target/release/glm5_run.exe '+%F %T') head $(git rev-parse --short HEAD) dirty $(git status --porcelain -- . | wc -l) N $N all-arms: ${ALL[*]}" | tee -a $D/run.log
export CROW_CNQ='C:\Users\robin\dev\crow-nest\converter\GLM-5.3-Flash-MUL1K3.cnq'
TOK='C:\Users\robin\dev\crow-nest\models\GLM-5.3-Flash-original\tokenizer.json'; export CROW_TOKENIZER="$TOK"
export CROW_GLM_MTP_OVERLAY='C:\Users\robin\dev\crow-nest\converter\GLM-5.3-Flash-MTP-overlay.cnq'
ARM3="$(cat $R/runs/glm53-flash/quick/arm3.env)"
NOLANE="$(echo "$ARM3" | tr ' ' '\n' | grep -v '^CROW_GLM_CPU_LANE=' | tr '\n' ' ')"
run_arm() {  # label, env...
  local L=$1; shift
  echo "== $L start $(date +%T) vram_used_mib $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) env-extra: $*" | tee -a $D/run.log
  env -u CROW_GLM_MTP -u CROW_GLM_KV_LAZY "$@" timeout 1800 target/release/glm5_run --cold --reps 1 --prompt-ids $D/req2.ids -n $N \
      --vram-slots 37 --tokenizer "$TOK" --json $D/$L.json 2>&1 | sed -E 's/  \[l[0-9]+ [0-9/ l]*\]$//' > $D/$L.log
  echo "== $L rc=${PIPESTATUS[0]} done $(date +%T)" | tee -a $D/run.log
  grep -hE "rep 1/1 .*(prefill:|decode:)|decode MTP:|refused|KV after rep" $D/$L.log | tee -a $D/run.log
}
run_arm A $ARM3 "${ALL[@]}"
run_arm B $NOLANE "${ALL[@]}"
run_arm C $NOLANE CROW_GLM_MTP=1 "${ALL[@]}"
echo "all done $(date +%T)" | tee -a $D/run.log
