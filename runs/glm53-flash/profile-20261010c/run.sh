#!/bin/bash
# Nsight profiles of the current state (glm-integration 679d3eb, ARM), robin's go 2026-10-10. Diagnosis only.
W=/c/Users/robin/dev/crow-nest-wt-int; D=$W/runs/glm53-flash/profile-20261010c; E=$W/engine
export CROW_CNQ='C:\Users\robin\dev\crow-nest\converter\GLM-5.3-Flash-MUL1K3.cnq'
export CROW_TOKENIZER='C:\Users\robin\dev\crow-nest\models\GLM-5.3-Flash-original\tokenizer.json'
NSYS="/c/Program Files/NVIDIA Corporation/Nsight Systems 2026.5.1/target-windows-x64/nsys.exe"
ARM="CROW_NVME_POOL=1 CROW_NVME_POOL_THREADS=16 CROW_GLM_HCFUSE=1 CROW_GLM_DENSE_GEMM=1 CROW_GLM_CPU_LANE=split CROW_PINNED_ALLOC=host CROW_GLM_ARENA=global CROW_GLM_ARENA_WARM=$W/runs/glm53-flash/arena-warm/glm53-cal4-decode-counts.json CROW_GLM_ARENA_ELASTIC_GB=10 CROW_GLM_ARENA_STAGE_GB=2.6 CROW_CHUNK=8192 CROW_GLM_FLAGS=1 CROW_GLM_STAGER=1 CROW_GLM_PREFETCH=1 CROW_GLM_PREFETCH_SIDE=1 CROW_GLM_SHARED_OVERLAP=1 CROW_GLM_CONTROLLER=1 CROW_GLM_LA=1"
cd $E
echo "start $(date) head $(git rev-parse --short HEAD) builds $(tasklist | grep -ciE 'cargo|rustc')"
for P in "prefill|--random-ids 3 --prompt-tokens 8192 -n 4" "decode|--random-ids 5 --prompt-tokens 64 -n 96"; do
  IFS='|' read -r N A <<< "$P"
  env $ARM "$NSYS" profile --trace=cuda --sample=none --force-overwrite true -o $D/$N target/release/glm5_run.exe --reps 1 $A --json $D/$N.json > $D/$N.log 2>&1; echo "$N nsys rc=$?"
  "$NSYS" stats --force-export=true --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum --format csv -o $D/${N}_stats $D/$N.nsys-rep >> $D/$N.log 2>&1; echo "$N stats rc=$?"
done
echo "done $(date)"
