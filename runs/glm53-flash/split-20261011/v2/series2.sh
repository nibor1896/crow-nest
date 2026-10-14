#!/bin/bash
# #213 step 1 (corrected): --vram-slots 37, -n 256 --reps 3 (rep 1 cold, reps 2-3 warm), arms alternated, n=2 processes per arm.
# Takes C:\Users\robin\dev\.gpu-lock itself (waits for series 1 to release it), removes it on exit. DRY=1 prints only.
W=/c/Users/robin/dev; WT=$W/crow-nest-wt-int; Q=$W/crow-nest/runs/glm53-flash/quick
OUT=$W/crow-nest/runs/glm53-flash/split-20261011/v2; mkdir -p $OUT
E=$WT/engine; BIN=$E/target/release/glm5_run.exe
if [ -z "$DRY" ]; then
  until mkdir $W/.gpu-lock 2>/dev/null; do sleep 5; done
  trap 'rmdir $W/.gpu-lock && echo "lock removed $(date +%T)" | tee -a $OUT/series.log' EXIT
fi
export CROW_CNQ='C:\Users\robin\dev\crow-nest\converter\GLM-5.3-Flash-MUL1K3.cnq'
TOK='C:\Users\robin\dev\crow-nest\models\GLM-5.3-Flash-original\tokenizer.json'; export CROW_TOKENIZER="$TOK"
ARM="$(cat $Q/arm2.env) CROW_GLM_RT2=1 $(cat $Q/arm3.env)"
TXT="Explain in detail how a modern CPU cache hierarchy works: L1, L2, L3, coherence protocols, write-back versus write-through, and how false sharing hurts multithreaded code. Give concrete examples."
REQ2=$W/crow-nest/runs/glm53-flash/split-20261011/req2-glm5-model.ids
run() { # label rep extra-env...
  local L=$1 R=$2; shift 2; local D=$OUT/$L-r$R; mkdir -p $D; local P
  case $L in S) P=(--tokenizer "$TOK" --prompt "$TXT");; LS) P=(--prompt-tokens 4096);; *) P=(--prompt-ids $REQ2);; esac
  echo "== $L r$R $(date +%T) bin $(date -r $BIN +'%F %T') extra: $*" | tee -a $OUT/series.log
  if [ -n "$DRY" ]; then echo env ARM "$@" timeout 600 $BIN --vram-slots 37 --reps 3 "${P[@]}" -n 256 --json $D/decode.json; return; fi
  (cd $E && env $ARM "$@" timeout 600 $BIN --vram-slots 37 --reps 3 "${P[@]}" -n 256 --json $D/decode.json > $D/decode.log 2>&1); echo "rc=$?" | tee -a $OUT/series.log
  grep -hE "rep [0-9]/3 .*decode: " $D/decode.log | sed -E 's/.*(rep [0-9]\/3).*decode: /\1 decode: /' | tee -a $OUT/series.log
}
arm() { case $1 in S|LS|LR) run $1 $2;; LR-lazy0) run $1 $2 CROW_GLM_ARENA_LAZY_REFILL=0;; LR-pc0) run $1 $2 CROW_GLM_ARENA_PROMPT_COUNT=0;; LR-lend0) run $1 $2 CROW_GLM_ARENA_STAGE_LEND=0;; esac; }
echo "series v2 start $(date +%T); procs: $(tasklist | grep -ciE 'glm5_run|serve.exe|llama|cargo|rustc')" | tee -a $OUT/series.log
for a in S LS LR LR-lazy0 LR-pc0 LR-lend0; do arm $a 1; done
for a in LR-lend0 LR-pc0 LR-lazy0 LR LS S; do arm $a 2; done
echo "series v2 done $(date +%T)" | tee -a $OUT/series.log
