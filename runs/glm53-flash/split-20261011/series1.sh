#!/bin/bash
# Decode-after-long-prompt split, 2026-10-11. Lock C:\Users\robin\dev\.gpu-lock is taken by the caller; removed on exit.
# DRY=1 prints the commands only.
W=/c/Users/robin/dev; WT=$W/crow-nest-wt-int; Q=$W/crow-nest/runs/glm53-flash/quick
OUT=$W/crow-nest/runs/glm53-flash/split-20261011; mkdir -p $OUT
E=$WT/engine; BIN=$E/target/release/glm5_run.exe
[ -z "$DRY" ] && trap 'rmdir $W/.gpu-lock && echo "lock removed $(date +%T)" | tee -a $OUT/series.log' EXIT
export CROW_CNQ='C:\Users\robin\dev\crow-nest\converter\GLM-5.3-Flash-MUL1K3.cnq'
TOK='C:\Users\robin\dev\crow-nest\models\GLM-5.3-Flash-original\tokenizer.json'; export CROW_TOKENIZER="$TOK"
ARM="$(cat $Q/arm2.env) CROW_GLM_RT2=1 $(cat $Q/arm3.env)"   # as bundle.sh -> quick.sh applies arm3.env
TXT="Explain in detail how a modern CPU cache hierarchy works: L1, L2, L3, coherence protocols, write-back versus write-through, and how false sharing hurts multithreaded code. Give concrete examples."
LONG="--prompt-ids $Q/prefill8192.ids --prompt-tokens 4096"
run() { # label rep extra-env...
  local L=$1 R=$2; shift 2; local D=$OUT/$L-r$R; mkdir -p $D
  local P; if [ $L = A0 ]; then P=(--tokenizer "$TOK" --prompt "$TXT"); else P=($LONG); fi
  echo "== $L r$R $(date +%T) bin $(date -r $BIN +'%F %T') extra: $*" | tee -a $OUT/series.log
  if [ -n "$DRY" ]; then echo env ARM "$@" timeout 300 $BIN --cold --reps 1 "${P[@]}" -n 256 --json $D/decode.json; return; fi
  (cd $E && env $ARM "$@" timeout 300 $BIN --cold --reps 1 "${P[@]}" -n 256 --json $D/decode.json > $D/decode.log 2>&1); echo "rc=$?" | tee -a $OUT/series.log
  grep -hE "rep 1/1 .*decode: " $D/decode.log | sed -E 's/.*decode: /decode: /' | tee -a $OUT/series.log
}
arm() { case $1 in A0|A1) run $1 $2;; A2) run A2 $2 CROW_GLM_ARENA_LAZY_REFILL=0;; A3) run A3 $2 CROW_GLM_ARENA_PROMPT_COUNT=0;; A4) run A4 $2 CROW_GLM_ARENA_STAGE_LEND=0;; esac; }
echo "series start $(date +%T); procs: $(tasklist | grep -ciE 'glm5_run|serve.exe|llama|cargo|rustc')" | tee -a $OUT/series.log
for a in A0 A1 A2 A3 A4; do arm $a 1; done
for a in A4 A3 A2 A1 A0; do arm $a 2; done
echo "series done $(date +%T)" | tee -a $OUT/series.log
