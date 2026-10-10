#!/bin/bash
# #188 CROW_GLM_ARENA_PROMPT_COUNT smoke: the quick.sh decode command (operating set: arm2.env + RT2 + the twelve
# switches of quick/b5-all12), -n 32, switch off / on; ids, NVMe experts and VRAM hits per token from the JSON.
# -nolane: the same with CROW_GLM_CPU_LANE=0 (the GPU reads VRAM and pinned records with the same bits, so the ids
# cannot depend on where a record lies; under split the CPU lane's combos have other bits by design).
# Usage: smoke.sh off|on|off-nolane|on-nolane   (hold C:\Users\robin\dev\.gpu-lock around it: locked.sh)
W=/c/Users/robin/dev/crow-nest-wt-lanes; Q=/c/Users/robin/dev/crow-nest-wt-int/runs/glm53-flash/quick; D=$W/runs/glm53-flash/prompt-count-20261010/smoke
export CROW_CNQ='C:\Users\robin\dev\crow-nest\converter\GLM-5.3-Flash-MUL1K3.cnq'
TOK='C:\Users\robin\dev\crow-nest\models\GLM-5.3-Flash-original\tokenizer.json'; export CROW_TOKENIZER="$TOK"
ARM="$(cat $Q/arm2.env) CROW_GLM_RT2=1 CROW_GLM_ARENA_FREQ=1 CROW_GLM_ARENA_REGROW=1 CROW_GLM_EMBED_GATHER=1 CROW_GLM_RT2_TABLE_SM=1 CROW_GLM_SIDE_NOJOIN=1 CROW_NVME_DEMAND_FIRST=1 CROW_GLM_GUESS_TRIM=1 CROW_GLM_ARENA_STAGE_LEND=1 CROW_GLM_ARENA_LAZY_REFILL=1 CROW_GLM_MUL1_FUSE=1 CROW_GLM_ATTN_FUSE=1 CROW_GLM_RT2_GATHER_EARLY=1"
P="Explain in detail how a modern CPU cache hierarchy works: L1, L2, L3, coherence protocols, write-back versus write-through, and how false sharing hurts multithreaded code. Give concrete examples."
LANE=""
case "$1" in
  off) PC=0;; on) PC=1;;
  off-nolane) PC=0; LANE="CROW_GLM_CPU_LANE=0";; on-nolane) PC=1; LANE="CROW_GLM_CPU_LANE=0";;
  *) echo "usage: smoke.sh off|on|off-nolane|on-nolane"; exit 2;;
esac
mkdir -p $D
cd $W/engine
env $ARM $LANE CROW_GLM_ARENA_PROMPT_COUNT=$PC timeout 150 target/release/glm5_run --cold --reps 1 --tokenizer "$TOK" --prompt "$P" -n 32 --json $D/$1.json > $D/$1.log 2>&1; echo "$1 rc=$?"
