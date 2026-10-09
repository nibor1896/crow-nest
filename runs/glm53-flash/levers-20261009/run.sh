#!/bin/bash
# Lever block 2026-10-09: every lever alone and two combinations vs the baseline, two passes (forward, reverse).
cd /c/Users/robin/dev/crow-nest/engine
D=../runs/glm53-flash/levers-20261009
ARMS=(
"base||"
"ioring2|CROW_NVME_BACKEND=ioring|--readers 2"
"flags|CROW_GLM_FLAGS=1|"
"lookahead|CROW_GLM_LOOKAHEAD=1|"
"zerocopy|CROW_GLM_PINNED=zerocopy|"
"host|CROW_PINNED_ALLOC=host|"
"host_zerocopy|CROW_PINNED_ALLOC=host CROW_GLM_PINNED=zerocopy|"
"host_cpulane|CROW_PINNED_ALLOC=host CROW_GLM_CPU_LANE=1|"
"all_zerocopy|CROW_NVME_BACKEND=ioring CROW_GLM_FLAGS=1 CROW_GLM_LOOKAHEAD=1 CROW_GLM_PINNED=zerocopy|--readers 2"
"all_cpulane|CROW_NVME_BACKEND=ioring CROW_GLM_FLAGS=1 CROW_GLM_LOOKAHEAD=1 CROW_PINNED_ALLOC=host CROW_GLM_CPU_LANE=1|--readers 2"
)
run() { local pass=$1 spec=$2; IFS='|' read -r name envs args <<< "$spec"; local L=$D/p$pass-$name.log
  { echo "start $(date) head $(git rev-parse --short HEAD) env: $envs args: $args"; echo "builds running: $(tasklist | grep -ciE 'cargo|rustc')"
    env $envs target/release/glm5_run -n 64 --reps 3 $args --json $D/p$pass-$name.json; echo "rc=$? end $(date)"; echo "builds running: $(tasklist | grep -ciE 'cargo|rustc')"; } > $L 2>&1; }
for a in "${ARMS[@]}"; do run 1 "$a"; done
for (( i=${#ARMS[@]}-1; i>=0; i-- )); do run 2 "${ARMS[$i]}"; done
echo "block done $(date)"
