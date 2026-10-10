#!/bin/bash
# exp1: serve FULL (as sweep2) answering one fixed short request twice, then glm5_run FULL on serve's prompt ids
W=/c/Users/robin/dev/crow-nest-wt-int; E=$W/engine; S=$(dirname "$0"); S=$(cd "$S" && pwd)
export CROW_CNQ='C:\Users\robin\dev\crow-nest\converter\GLM-5.3-Flash-MUL1K3.cnq'
TOK='C:\Users\robin\dev\crow-nest\models\GLM-5.3-Flash-original\tokenizer.json'; export CROW_TOKENIZER="$TOK"
FULL="CROW_NVME_POOL=1 CROW_GLM_HCFUSE=1 CROW_GLM_CPU_LANE=split CROW_PINNED_ALLOC=host CROW_GLM_ARENA=global CROW_GLM_ARENA_WARM=$W/runs/glm53-flash/arena-warm/glm53-cal4-decode-counts.json CROW_GLM_ARENA_ELASTIC_GB=10 CROW_GLM_ARENA_STAGE_GB=2.6 CROW_CHUNK=8192 CROW_GLM_FLAGS=1 CROW_GLM_STAGER=1 CROW_GLM_PREFETCH=1 CROW_GLM_PREFETCH_SIDE=1 CROW_GLM_SHARED_OVERLAP=1 CROW_GLM_CONTROLLER=1 CROW_GLM_LA=1"
TAG=${1:-full}; XENV=${2:-}
cd $E
echo "start $(date) builds $(tasklist | grep -ciE 'cargo|rustc|nvcc')"
env $FULL $XENV CROW_GLM_MAX_BATCH=1 CROW_LOG=info,chat=debug target/release/serve.exe --port 8099 > $S/serve-$TAG.log 2>&1 & SP=$!
ok=0; for i in $(seq 1 120); do sleep 5; curl -sf http://127.0.0.1:8099/health >/dev/null && { ok=1; break; }; kill -0 $SP 2>/dev/null || break; done
echo "serve boot ok=$ok"
if [ $ok = 1 ]; then
  python -I $S/req.py http://127.0.0.1:8099 $S/r1-$TAG.json 200
  python -I $S/req.py http://127.0.0.1:8099 $S/r2-$TAG.json 200
fi
taskkill //PID $SP //F >/dev/null 2>&1; kill $SP 2>/dev/null; sleep 8
tasklist | grep -i serve.exe && echo "SERVE STILL ALIVE"
grep -o '\[chat\] prompt ids \[[0-9, ]*\]' $S/serve-$TAG.log | head -1 | sed 's/.*ids //' > $S/ids-$TAG.txt
if [ -s $S/ids-$TAG.txt ] && [ "$3" != "noglm" ]; then
  env $FULL $XENV target/release/glm5_run.exe --prompt-ids $S/ids-$TAG.txt -n 200 --reps 2 --tokenizer "$TOK" --json $S/glm5run-$TAG.json > $S/glm5run-$TAG.log 2>&1
  echo "glm5_run rc=$?"
fi
echo "done $(date)"
