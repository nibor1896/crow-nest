#!/bin/bash
# Prefill pair re-run: CROW_CHUNK=32 books the pageable prefill landing off the pinned budget, so the plan
# allows 121 pinned slots, not 123 (first try refused, p1/p2-pre_chunk32.log). Both arms at 121.
cd /c/Users/robin/dev/crow-nest/engine
D=../runs/glm53-flash/block-20261010
T="--vram-slots 36 --pinned-slots 121"
Z="CROW_GLM_PINNED=zerocopy"
PRE=("pre121_chunk1|$Z CROW_CHUNK=1|" "pre121_chunk32|$Z CROW_CHUNK=32|")
run() { local pass=$1 spec=$2 extra=$3; IFS='|' read -r name envs args <<< "$spec"; local L=$D/p$pass-$name.log
  { echo "start $(date) head $(git rev-parse --short HEAD) env: $envs args: $T $args $extra"; echo "builds running: $(tasklist | grep -ciE 'cargo|rustc')"
    env $envs target/release/glm5_run $T $args $extra --json $D/p$pass-$name.json; echo "rc=$? end $(date)"; echo "builds running: $(tasklist | grep -ciE 'cargo|rustc')"; } > $L 2>&1; }
for a in "${PRE[@]}"; do run 1 "$a" "-n 8 --reps 1 --prompt-tokens 2048"; done
for (( i=${#PRE[@]}-1; i>=0; i-- )); do run 2 "${PRE[$i]}" "-n 8 --reps 1 --prompt-tokens 2048"; done
# Nsight profile of the fastest decode arm (stager: zerocopy + flags + stager), same tiers as the block
P=../runs/glm53-flash/profile-20261010; mkdir -p $P
NSYS="/c/Program Files/NVIDIA Corporation/Nsight Systems 2026.5.1/target-windows-x64/nsys.exe"
env $Z CROW_GLM_FLAGS=1 CROW_GLM_STAGER=1 "$NSYS" profile --trace=cuda --sample=none -o $P/stager16 target/release/glm5_run.exe --vram-slots 36 --pinned-slots 123 -n 16 > $P/profile.log 2>&1; echo "nsys rc=$?" >> $P/profile.log
"$NSYS" stats --report cuda_gpu_kern_sum,cuda_api_sum --format csv -o $P/stats $P/stager16.nsys-rep >> $P/profile.log 2>&1; echo "stats rc=$?" >> $P/profile.log
echo "pre+profile done $(date)"
