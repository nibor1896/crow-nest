#!/bin/bash
# #196 Stage 3 prefill profile (glm-integration 28f459a, operating set arm2.env + CROW_GLM_RT2=1).
# The quick.sh prefill command verbatim, once unprofiled and once under nsys. Diagnosis only.
# Usage: run.sh <tag> [nsys|plain] [extra nsys args...]
W=/c/Users/robin/dev/crow-nest-wt-int; Q=$W/runs/glm53-flash/quick; D=$W/runs/glm53-flash/profile-20261010i-prefill; E=$W/engine
T=$1; MODE=$2; shift 2
export CROW_CNQ='C:\Users\robin\dev\crow-nest\converter\GLM-5.3-Flash-MUL1K3.cnq'
TOK='C:\Users\robin\dev\crow-nest\models\GLM-5.3-Flash-original\tokenizer.json'; export CROW_TOKENIZER="$TOK"
ARM="$(cat $Q/arm2.env) CROW_GLM_RT2=1"
NSYS="/c/Program Files/NVIDIA Corporation/Nsight Systems 2026.5.1/target-windows-x64/nsys.exe"
cd $E
echo "start $(date +%T) head $(git rev-parse --short HEAD) builds $(tasklist | grep -ciE 'cargo|rustc') tag $T mode $MODE nsys-args: $*" | tee -a $D/run.log
if [ "$MODE" = nsys ]; then
  env $ARM "$NSYS" profile --trace=cuda --sample=none --cpuctxsw=none "$@" --force-overwrite true -o $D/$T \
    target/release/glm5_run.exe --cold --reps 1 --prompt-ids $Q/prefill8192.ids -n 4 --json $D/$T.json > $D/$T.log 2>&1; echo "$T nsys rc=$?" | tee -a $D/run.log
else
  env $ARM timeout 150 target/release/glm5_run --cold --reps 1 --prompt-ids $Q/prefill8192.ids -n 4 --json $D/$T.json > $D/$T.log 2>&1; echo "$T rc=$?" | tee -a $D/run.log
fi
grep -hE "rep 1/1 .*prefill:" $D/$T.log | sed -E 's/.*(prefill): /\1: /' | tee -a $D/run.log
echo "done $(date +%T)" | tee -a $D/run.log
