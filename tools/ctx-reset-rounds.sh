#!/usr/bin/env bash
# #82 evidence rounds (2026-09-20): run ctx_reset_probe teardown orders, each in
# its own process, and sample MemAvailable + the nvidia_uvm refcount around every
# run. A mode that never returns is killed by timeout (rc 124/137) - its last
# "about to" line names the call that hung, and the "after" snapshot sizes the
# leak it left. Deliberately small tiers by default (2 GiB): a round must not
# cost the machine its clean boot.
#   tools/ctx-reset-rounds.sh [gib=2] [mode ...]   (default modes: free-exit reset-hold release-reset)
set -u
cd "$(dirname "$0")/.."
bin=engine/target/release/ctx_reset_probe
gib=${1:-2}
if [ "$#" -gt 0 ]; then shift; fi
if [ "$#" -eq 0 ]; then set -- free-exit reset-hold release-reset; fi

snap() {
  local m r
  m=$(free -m | awk '/^Mem:/{print $7}')
  r=$(awk '$1=="nvidia_uvm"{print $3}' /proc/modules)
  echo "MemAvailable ${m} MiB, nvidia_uvm refcount ${r}"
}

for mode in "$@"; do
  echo "=== ${mode} (${gib} GiB) before: $(snap)"
  timeout -k 5 90 "$bin" "$mode" "$gib" 96
  rc=$?
  sleep 2
  echo "=== ${mode} rc=${rc} after: $(snap)"
done
