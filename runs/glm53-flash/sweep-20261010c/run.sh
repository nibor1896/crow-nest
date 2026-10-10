#!/bin/bash
# Real-container measurement after the RCA fixes (glm-integration b85cd01), robin's go 2026-10-10. Cold only.
W=/c/Users/robin/dev/crow-nest-wt-int; D=$W/runs/glm53-flash/sweep-20261010c; E=$W/engine
export CROW_CNQ='C:\Users\robin\dev\crow-nest\converter\GLM-5.3-Flash-MUL1K3.cnq'
TOK='C:\Users\robin\dev\crow-nest\models\GLM-5.3-Flash-original\tokenizer.json'; export CROW_TOKENIZER="$TOK"
ARM="CROW_NVME_POOL=1 CROW_NVME_POOL_THREADS=16 CROW_GLM_HCFUSE=1 CROW_GLM_DENSE_GEMM=1 CROW_GLM_CPU_LANE=split CROW_PINNED_ALLOC=host CROW_GLM_ARENA=global CROW_GLM_ARENA_WARM=$W/runs/glm53-flash/arena-warm/glm53-cal4-decode-counts.json CROW_GLM_ARENA_ELASTIC_GB=10 CROW_GLM_ARENA_STAGE_GB=2.6 CROW_CHUNK=8192 CROW_GLM_FLAGS=1 CROW_GLM_STAGER=1 CROW_GLM_PREFETCH=1 CROW_GLM_PREFETCH_SIDE=1 CROW_GLM_SHARED_OVERLAP=1 CROW_GLM_CONTROLLER=1 CROW_GLM_LA=1"
cd $E
echo "start $(date) head $(git rev-parse --short HEAD) builds $(tasklist | grep -ciE 'cargo|rustc')"
# 1) ids vs default path (default ids from sweep-20261010/ids-default.json, same fixed prompt, 64 tokens), cold
env $ARM target/release/glm5_run -n 64 --cold --json $D/ids-arm.json > $D/ids-arm.log 2>&1; echo "ids-arm rc=$?"
# 2) cold decode with a fresh random prompt per rep (no warm-cache artifact)
env $ARM target/release/glm5_run -n 128 --reps 3 --random-ids 7 --prompt-tokens 64 --json $D/cold-arm.json > $D/cold-arm.log 2>&1; echo "cold-arm rc=$?"
# 3) template-comparable sweep against serve
env $ARM CROW_GLM_MAX_BATCH=1 target/release/serve.exe --port 8099 > $D/serve.log 2>&1 & SP=$!
ok=0; for i in $(seq 1 120); do sleep 5; curl -sf http://127.0.0.1:8099/health >/dev/null && { ok=1; break; }; kill -0 $SP 2>/dev/null || break; done
echo "serve boot ok=$ok after $((i*5)) s"
[ $ok = 1 ] && { python -I $W/tools/glm_sweep.py --url http://127.0.0.1:8099 --card rtx5090 --config "b85cd01 ARM, MAX_BATCH=1, pinned 46 GiB cap" --tokenizer "$TOK" --prefill 8192 32768 --conc 1 --reps 3 --dec-reps 2 --out $D/sweep.json > $D/sweep.log 2>&1; echo "sweep rc=$?"; }
taskkill //PID $SP //F >/dev/null 2>&1; kill $SP 2>/dev/null
echo "done $(date)"
