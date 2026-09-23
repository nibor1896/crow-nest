# Sourced helper (#103, 2026-09-23): gate an engine boot on the ENGINE's
# own host-RAM view instead of MemAvailable.
#
#   . "$root/tools/pin-room.sh"
#   pin_room <need_gib>    0 = free_for_pin >= need; 1 = a real shortage (live
#                          processes hold the RAM); 2 = ramcheck is not built
#
# Why not MemAvailable: every engine before this fix pinned its cold tier with a
# write-combined cuMemHostAlloc, and the NVIDIA driver parks those pages in its
# sysmem page pool after the free (kernel-open/nvidia/nv-vm.c). The pool is in no
# /proc/meminfo class, so MemAvailable misses it, yet it is reclaimable: the
# next allocation takes it (driver allocation from the pool, a registered tier
# through the pool's shrinker - 8 GiB went back in 3 s, 2026-09-23). A
# MemAvailable gate therefore printed "REBOOT needed" for RAM the next engine
# gets anyway (MEAS-0923 01:11, "pool flat at 44 GiB", need 51). `ramcheck`
# (engine/src/bin/ramcheck.rs) prints cuda::free_physical_ram_parts - the very
# figure the loader budgets with - and a refusal there names the processes that
# hold the RAM. No balloon, no reboot.
#
# A serve that is still exiting holds its tier until it is gone, so the check is
# retried for PIN_ROOM_WAIT_S seconds (default 120) before it says "short".
# RAMCHECK overrides the binary (default <repo>/engine/target/release/ramcheck).
pin_room() {
    local need=$1 t0=$SECONDS out rc
    local here; here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    local bin="${RAMCHECK:-$here/engine/target/release/ramcheck}"
    if [ ! -x "$bin" ]; then
        echo "  pin_room: $bin is not built - cd engine && cargo build --release --bin ramcheck"
        return 2
    fi
    while true; do
        out=$("$bin" --need "$need"); rc=$?
        if [ "$rc" -eq 0 ]; then
            echo "$out" | head -1 | sed 's/^/  pin_room: /'
            return 0
        fi
        if [ $((SECONDS - t0)) -ge "${PIN_ROOM_WAIT_S:-120}" ]; then
            echo "$out" | sed 's/^/  pin_room: /'
            return 1
        fi
        sleep 5
    done
}
