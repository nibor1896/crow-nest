#!/bin/bash
# Real-container measurement of the glm-flash-lite rebuild (glm-integration a923e79), robin's go 2026-10-10.
# 0) ids check: glm5_run default vs full arm (no batch), 64 tokens, same fixed prompt.
# 1) template-comparable sweep (tools/glm_sweep.py = bench/sweep.py at 6769b27) against serve with the full arm.
W=/c/Users/robin/dev/crow-nest-wt-int; D=$W/runs/glm53-flash/sweep-20261010; E=$W/engine
export CROW_CNQ='C:\Users\robin\dev\crow-nest\converter\GLM-5.3-Flash-MUL1K3.cnq'
TOK='C:\Users\robin\dev\crow-nest\models\GLM-5.3-Flash-original\tokenizer.json'; export CROW_TOKENIZER="$TOK"
FULL="CROW_NVME_POOL=1 CROW_GLM_HCFUSE=1 CROW_GLM_CPU_LANE=split CROW_PINNED_ALLOC=host CROW_GLM_ARENA=global CROW_GLM_ARENA_WARM=$W/runs/glm53-flash/arena-warm/glm53-cal4-decode-counts.json CROW_GLM_ARENA_ELASTIC_GB=10 CROW_GLM_ARENA_STAGE_GB=2.6 CROW_CHUNK=8192 CROW_GLM_FLAGS=1 CROW_GLM_STAGER=1 CROW_GLM_PREFETCH=1 CROW_GLM_PREFETCH_SIDE=1 CROW_GLM_SHARED_OVERLAP=1 CROW_GLM_CONTROLLER=1 CROW_GLM_LA=1"
cd $E
echo "start $(date) head $(git rev-parse --short HEAD) builds $(tasklist | grep -ciE 'cargo|rustc')"
: # ids-default done in run 1
: # ids-full done by the fix agent
for MB in 1; do
  env $FULL CROW_GLM_MAX_BATCH=$MB target/release/serve.exe --port 8099 > $D/serve2-mb$MB.log 2>&1 & SP=$!
  ok=0; for i in $(seq 1 120); do sleep 5; curl -sf http://127.0.0.1:8099/health >/dev/null && { ok=1; break; }; kill -0 $SP 2>/dev/null || break; done
  echo "serve mb$MB boot ok=$ok after $((i*5)) s"
  if [ $ok = 1 ]; then
    CONC="1 2 4"; [ $MB = 1 ] && CONC="1"
    python -I $W/tools/glm_sweep.py --url http://127.0.0.1:8099 --card rtx5090 --config "a923e79 full arm, CROW_GLM_MAX_BATCH=$MB, pinned 46 GiB cap" --tokenizer "$TOK" --prefill 8192 --conc $CONC --reps 3 --dec-reps 2 --out $D/sweep2-mb$MB.json > $D/sweep2-mb$MB.log 2>&1; echo "sweep mb$MB rc=$?"
    taskkill //PID $SP //F >/dev/null 2>&1; kill $SP 2>/dev/null; sleep 5; break
  fi
  taskkill //PID $SP //F >/dev/null 2>&1; kill $SP 2>/dev/null; sleep 5
done
echo "done $(date)"
