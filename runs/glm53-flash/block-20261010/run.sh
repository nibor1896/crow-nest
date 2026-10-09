#!/bin/bash
# Clean block after #191/#149/#186/#192: every new switch alone on the zerocopy basis and all together,
# two passes (forward, reverse); then prefill chunk 1 vs 32 at 2048 prompt tokens.
# Equal tiers in every arm: MTP books ~3 GB VRAM and the stager 76 MB pinned, so all arms run
# --vram-slots 36 --pinned-slots 123 (prefill arms too, the chunk books VRAM).
cd /c/Users/robin/dev/crow-nest/engine
D=../runs/glm53-flash/block-20261010
T="--vram-slots 36 --pinned-slots 123"
Z="CROW_GLM_PINNED=zerocopy"
ARMS=(
"base|$Z|"
"graph|$Z CROW_GLM_GRAPH=1|"
"flags|$Z CROW_GLM_FLAGS=1|"
"stager|$Z CROW_GLM_FLAGS=1 CROW_GLM_STAGER=1|"
"mtp1|$Z CROW_GLM_MTP=1|"
"mtp2|$Z CROW_GLM_MTP=2|"
"all_mtp1|$Z CROW_GLM_GRAPH=1 CROW_GLM_FLAGS=1 CROW_GLM_STAGER=1 CROW_GLM_MTP=1|"
"all_mtp2|$Z CROW_GLM_GRAPH=1 CROW_GLM_FLAGS=1 CROW_GLM_STAGER=1 CROW_GLM_MTP=2|"
)
PRE=(
"pre_chunk1|$Z CROW_CHUNK=1|"
"pre_chunk32|$Z CROW_CHUNK=32|"
)
run() { local pass=$1 spec=$2 extra=$3; IFS='|' read -r name envs args <<< "$spec"; local L=$D/p$pass-$name.log
  { echo "start $(date) head $(git rev-parse --short HEAD) env: $envs args: $T $args $extra"; echo "builds running: $(tasklist | grep -ciE 'cargo|rustc')"
    env $envs target/release/glm5_run $T $args $extra --json $D/p$pass-$name.json; echo "rc=$? end $(date)"; echo "builds running: $(tasklist | grep -ciE 'cargo|rustc')"; } > $L 2>&1; }
for a in "${ARMS[@]}"; do run 1 "$a" "-n 64 --reps 3"; done
for (( i=${#ARMS[@]}-1; i>=0; i-- )); do run 2 "${ARMS[$i]}" "-n 64 --reps 3"; done
for a in "${PRE[@]}"; do run 1 "$a" "-n 8 --reps 1 --prompt-tokens 2048"; done
for (( i=${#PRE[@]}-1; i>=0; i-- )); do run 2 "${PRE[$i]}" "-n 8 --reps 1 --prompt-tokens 2048"; done
echo "block done $(date)"
