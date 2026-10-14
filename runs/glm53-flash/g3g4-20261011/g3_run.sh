#!/bin/bash
# G3 (per layer + head, `decode glmgolden`, #161/#165 form) with the operating set arm3.env, twice, 2026-10-11.
# Goldens: models/GLM-5.3-Flash-step06/ref-mul1-all (runner on the same 3-bit container, all 45 layers + head).
R=/c/Users/robin/dev/crow-nest; D=$R/runs/glm53-flash/g3g4-20261011; E=$R/engine
export CROW_CNQ='C:\Users\robin\dev\crow-nest\converter\GLM-5.3-Flash-MUL1K3.cnq'
ARM="$(cat $R/runs/glm53-flash/quick/arm3.env)"
cd $E
echo "start $(date +%T) head $(git rev-parse --short HEAD) builds $(tasklist | grep -ciE 'cargo|rustc') decode.exe $(date -r target/release/decode.exe +%FT%T)" | tee $D/g3.log
for k in 1 2; do
  env $ARM target/release/decode.exe glmgolden $R/models/GLM-5.3-Flash-step06/ref-mul1-all --cnq "$CROW_CNQ" > $D/g3-run$k.txt 2>&1
  echo "run $k rc=$? $(date +%T)" | tee -a $D/g3.log
  grep -E "glmgolden: |head top-1|logits anchor" $D/g3-run$k.txt | tee -a $D/g3.log
done
echo "done $(date +%T)" | tee -a $D/g3.log
