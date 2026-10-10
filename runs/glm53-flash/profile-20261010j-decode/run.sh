#!/bin/bash
# profile-20261010j-decode: the quick.sh decode command (ARM2 + CROW_GLM_RT2=1), glm-integration 28f459a,
# once unprofiled (plain/) and once under nsys (CUDA trace, no CPU sampling, no context switches).
W=/c/Users/robin/dev/crow-nest-wt-int; Q=$W/runs/glm53-flash/quick; D=$W/runs/glm53-flash/profile-20261010j-decode
export CROW_CNQ='C:\Users\robin\dev\crow-nest\converter\GLM-5.3-Flash-MUL1K3.cnq'
TOK='C:\Users\robin\dev\crow-nest\models\GLM-5.3-Flash-original\tokenizer.json'; export CROW_TOKENIZER="$TOK"
ARM="$(cat $Q/arm2.env) CROW_GLM_RT2=1"
NSYS="/c/Program Files/NVIDIA Corporation/Nsight Systems 2026.5.1/target-windows-x64/nsys.exe"
P="Explain in detail how a modern CPU cache hierarchy works: L1, L2, L3, coherence protocols, write-back versus write-through, and how false sharing hurts multithreaded code. Give concrete examples."
cd $W/engine
case "$1" in
plain) env $ARM timeout 120 target/release/glm5_run --cold --reps 1 --tokenizer "$TOK" --prompt "$P" -n 128 --json $D/plain.json > $D/plain.log 2>&1; echo "plain rc=$?";;
nsys) env $ARM "$NSYS" profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --force-overwrite=true -o $D/decode \
        target/release/glm5_run --cold --reps 1 --tokenizer "$TOK" --prompt "$P" -n 128 --json $D/decode.json > $D/decode.log 2>&1; echo "nsys rc=$?";;
esac
