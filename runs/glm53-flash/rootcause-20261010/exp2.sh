#!/bin/bash
# exp2: glm5_run FULL on the fixed prompt, -n $1 --reps 3 (the 10.7-11.7 claim), extra env $2
W=/c/Users/robin/dev/crow-nest-wt-int; E=$W/engine; S=$(cd "$(dirname "$0")" && pwd)
export CROW_CNQ='C:\Users\robin\dev\crow-nest\converter\GLM-5.3-Flash-MUL1K3.cnq'
TOK='C:\Users\robin\dev\crow-nest\models\GLM-5.3-Flash-original\tokenizer.json'
FULL="CROW_NVME_POOL=1 CROW_GLM_HCFUSE=1 CROW_GLM_CPU_LANE=split CROW_PINNED_ALLOC=host CROW_GLM_ARENA=global CROW_GLM_ARENA_WARM=$W/runs/glm53-flash/arena-warm/glm53-cal4-decode-counts.json CROW_GLM_ARENA_ELASTIC_GB=10 CROW_GLM_ARENA_STAGE_GB=2.6 CROW_CHUNK=8192 CROW_GLM_FLAGS=1 CROW_GLM_STAGER=1 CROW_GLM_PREFETCH=1 CROW_GLM_PREFETCH_SIDE=1 CROW_GLM_SHARED_OVERLAP=1 CROW_GLM_CONTROLLER=1 CROW_GLM_LA=1"
N=$1; TAG=$3; shift 3
cd $E; echo "start $(date) builds $(tasklist | grep -ciE 'cargo|rustc|nvcc')"
env $FULL $XENV target/release/glm5_run.exe -n $N --reps 3 --tokenizer "$TOK" "$@" > $S/glm5run-$TAG.log 2>&1
echo "rc=$? done $(date)"
