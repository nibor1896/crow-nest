#!/bin/bash
# run smoke.sh for each argument while holding <home>\dev\.gpu-lock (waits for it; removes only its own)
L=<home>/dev/.gpu-lock; H=$(dirname "$0")
until mkdir "$L" 2>/dev/null; do sleep 15; done
echo "lock taken $(date +%T)"
for a in "$@"; do "$H/smoke.sh" "$a"; done
rmdir "$L" && echo "lock released $(date +%T)"
