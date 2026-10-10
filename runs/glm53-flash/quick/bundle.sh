#!/bin/bash
# One 3-minute check per arm (quick.sh), each switch alone against ARM2 + RT2, under the GPU lock.
# Usage: bundle.sh <prefix> "<label>|<engine dir>|<extra env...>" ...
W=/c/Users/robin/dev; Q=$W/crow-nest-wt-int/runs/glm53-flash/quick; P=$1; shift
until mkdir $W/.gpu-lock 2>/dev/null; do sleep 10; done
trap 'rmdir $W/.gpu-lock' EXIT
for arm in "$@"; do
  IFS='|' read -r label eng extra <<< "$arm"
  echo "== $P-$label engine $eng ($(date -r $eng/target/release/glm5_run.exe +%T)) $extra"
  ENGINE_DIR=$eng bash $Q/quick.sh $P-$label $extra
done
