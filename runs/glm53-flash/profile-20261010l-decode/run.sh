#!/bin/bash
# profile-20261010l-decode: the quick.sh decode command (ARM2 + CROW_GLM_RT2=1 + the twelve switches of quick/b5-all12),
# glm-integration 3de5eff, twice unprofiled (plain, plain2) and once under nsys (CUDA trace, no CPU sampling, no context switches).
# Usage: run.sh plain|plain2|nsys   (hold C:\Users\robin\dev\.gpu-lock around it)
W=/c/Users/robin/dev/crow-nest-wt-int; Q=$W/runs/glm53-flash/quick; D=$W/runs/glm53-flash/profile-20261010l-decode
export CROW_CNQ='C:\Users\robin\dev\crow-nest\converter\GLM-5.3-Flash-MUL1K3.cnq'
TOK='C:\Users\robin\dev\crow-nest\models\GLM-5.3-Flash-original\tokenizer.json'; export CROW_TOKENIZER="$TOK"
ARM="$(cat $Q/arm2.env) CROW_GLM_RT2=1 CROW_GLM_ARENA_FREQ=1 CROW_GLM_ARENA_REGROW=1 CROW_GLM_EMBED_GATHER=1 CROW_GLM_RT2_TABLE_SM=1 CROW_GLM_SIDE_NOJOIN=1 CROW_NVME_DEMAND_FIRST=1 CROW_GLM_GUESS_TRIM=1 CROW_GLM_ARENA_STAGE_LEND=1 CROW_GLM_ARENA_LAZY_REFILL=1 CROW_GLM_MUL1_FUSE=1 CROW_GLM_ATTN_FUSE=1 CROW_GLM_RT2_GATHER_EARLY=1"
NSYS="/c/Program Files/NVIDIA Corporation/Nsight Systems 2026.5.1/target-windows-x64/nsys.exe"
P="Explain in detail how a modern CPU cache hierarchy works: L1, L2, L3, coherence protocols, write-back versus write-through, and how false sharing hurts multithreaded code. Give concrete examples."
cd $W/engine
case "$1" in
plain|plain2) env $ARM "${@:2}" timeout 120 target/release/glm5_run --cold --reps 1 --tokenizer "$TOK" --prompt "$P" -n 128 --json $D/$1.json > $D/$1.log 2>&1; echo "$1 rc=$?";;
nsys) env $ARM "${@:2}" "$NSYS" profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --force-overwrite=true -o $D/decode \
        target/release/glm5_run --cold --reps 1 --tokenizer "$TOK" --prompt "$P" -n 128 --json $D/decode.json > $D/decode.log 2>&1; echo "nsys rc=$?"
      "$NSYS" export --type sqlite --force-overwrite=true -o $D/decode.sqlite $D/decode.nsys-rep > /dev/null 2>&1; echo "export rc=$?";;
esac
grep -hE "decode:" $D/${1/nsys/decode}.log | tail -1
