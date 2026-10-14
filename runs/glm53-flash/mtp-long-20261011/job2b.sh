#!/bin/bash
# Follow-up of job2.sh: arm C (MTP) was refused at --vram-slots 37 ("above the #159 plan's 29" with the MTP overlay).
# B29 / C29 = arms B / C at --vram-slots 29 (the MTP plan), same prompt and N; caller holds the GPU lock.
W=/c/Users/robin/dev; R=$W/crow-nest; D=$R/runs/glm53-flash/mtp-long-20261011; E=$R/engine; N=${1:-5500}
cd $E || exit 1
export CROW_CNQ='C:\Users\robin\dev\crow-nest\converter\GLM-5.3-Flash-MUL1K3.cnq'
TOK='C:\Users\robin\dev\crow-nest\models\GLM-5.3-Flash-original\tokenizer.json'; export CROW_TOKENIZER="$TOK"
export CROW_GLM_MTP_OVERLAY='C:\Users\robin\dev\crow-nest\converter\GLM-5.3-Flash-MTP-overlay.cnq'
NOLANE="$(cat $R/runs/glm53-flash/quick/arm3.env | tr ' ' '\n' | grep -v '^CROW_GLM_CPU_LANE=' | tr '\n' ' ')"
run_arm() { local L=$1; shift
  echo "== $L start $(date +%T) vram_used_mib $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) --vram-slots 29 extra: $*" | tee -a $D/run.log
  env -u CROW_GLM_MTP -u CROW_GLM_KV_LAZY $NOLANE "$@" timeout 1800 target/release/glm5_run --cold --reps 1 --prompt-ids $D/req2.ids -n $N \
      --vram-slots 29 --tokenizer "$TOK" --json $D/$L.json 2>&1 | sed -E 's/  \[l[0-9]+ [0-9/ l]*\]$//' > $D/$L.log
  echo "== $L rc=${PIPESTATUS[0]} done $(date +%T)" | tee -a $D/run.log
  grep -hE "rep 1/1 .*(prefill:|decode:)|decode MTP:|refused" $D/$L.log | tee -a $D/run.log; }
run_arm B29
run_arm C29 CROW_GLM_MTP=1
echo "job2b done $(date +%T)" | tee -a $D/run.log
