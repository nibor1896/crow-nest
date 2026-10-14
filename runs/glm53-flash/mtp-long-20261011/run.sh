#!/bin/bash
# #192 follow-up, 2026-10-11 (robin's GPU go): does MTP pay off over a LONG decode on the operating set?
# Three arms, one long cold run each, same prompt (serve request 2 of runs/glm53-flash/g3g4-20261011: docs/glm5-model.md
# [0:12000] + question, chat template, 3,985 ids = req2.ids), same -n, --vram-slots 37, greedy, order A B C:
#   A = arm3.env as is (CROW_GLM_CPU_LANE=split), MTP off   (the operating point)
#   B = arm3.env without CROW_GLM_CPU_LANE, MTP off
#   C = arm3.env without CROW_GLM_CPU_LANE, CROW_GLM_MTP=1  (MTP refuses the CPU lane: glm5_tiers.rs spec_check)
# Waits for C:\Users\robin\dev\.gpu-lock and for no glm5_run/serve/decode process (poll 60 s), holds the lock for the
# whole job (bundle.sh convention), rebuilds glm5_run only if older than the engine sources.
# Usage: run.sh [N]   (default N 5500)
W=/c/Users/robin/dev; R=$W/crow-nest; D=$R/runs/glm53-flash/mtp-long-20261011; E=$R/engine; N=${1:-5500}
LOCK=$W/.gpu-lock
busy() { tasklist 2>/dev/null | grep -qiE "^(glm5_run|serve|decode)\.exe"; }
echo "wait start $(date +%T)" | tee -a $D/run.log
while :; do
  if [ ! -d $LOCK ] && ! busy; then
    if mkdir $LOCK 2>/dev/null; then break; fi
  fi
  sleep 60
done
trap 'rmdir $LOCK && echo "lock removed $(date +%T)" | tee -a $D/run.log' EXIT
echo "lock taken $(date +%T)" | tee -a $D/run.log
cd $E || exit 1
if [ -n "$(find src Cargo.toml build.rs -newer target/release/glm5_run.exe -type f 2>/dev/null | head -1)" ]; then
  echo "build glm5_run $(date +%T)" | tee -a $D/run.log
  cargo build --release --bin glm5_run > $D/build.log 2>&1 || { echo "build failed" | tee -a $D/run.log; exit 1; }
fi
echo "binary $(date -r target/release/glm5_run.exe '+%F %T') head $(git rev-parse --short HEAD) dirty $(git status --porcelain -- . | wc -l)" | tee -a $D/run.log
export CROW_CNQ='C:\Users\robin\dev\crow-nest\converter\GLM-5.3-Flash-MUL1K3.cnq'
TOK='C:\Users\robin\dev\crow-nest\models\GLM-5.3-Flash-original\tokenizer.json'; export CROW_TOKENIZER="$TOK"
export CROW_GLM_MTP_OVERLAY='C:\Users\robin\dev\crow-nest\converter\GLM-5.3-Flash-MTP-overlay.cnq'
ARM3="$(cat $R/runs/glm53-flash/quick/arm3.env)"
NOLANE="$(echo "$ARM3" | tr ' ' '\n' | grep -v '^CROW_GLM_CPU_LANE=' | tr '\n' ' ')"
run_arm() {  # label, env...
  local L=$1; shift
  echo "== $L start $(date +%T) vram_used_mib $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) env-extra: $*" | tee -a $D/run.log
  env -u CROW_GLM_MTP "$@" timeout 1800 target/release/glm5_run --cold --reps 1 --prompt-ids $D/req2.ids -n $N \
      --vram-slots 37 --tokenizer "$TOK" --json $D/$L.json 2>&1 | sed -E 's/  \[l[0-9]+ [0-9/ l]*\]$//' > $D/$L.log
  echo "== $L rc=${PIPESTATUS[0]} done $(date +%T)" | tee -a $D/run.log
  grep -hE "rep 1/1 .*(prefill:|decode:)|decode MTP:|refused" $D/$L.log | tee -a $D/run.log
}
run_arm A $ARM3
run_arm B $NOLANE
run_arm C $NOLANE CROW_GLM_MTP=1
echo "all done $(date +%T)" | tee -a $D/run.log
