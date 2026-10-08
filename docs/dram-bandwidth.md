# DRAM read bandwidth per core class

`probes/src/bin/p17_dram_bw.rs` is step 4 of the GLM-5.3-Flash series (issue #170, plan root #169).
It replaces the nominal 89.6 GB/s (2 channels x 5600 MT/s x 8 B, Core Ultra 9 285K with 2x 32 GB
DDR5-5600) with a measured streaming-read rate for three thread sets: all cores, the P cores, the
E cores. It changes nothing in the engine, the converter or the container.

> **Status:** the tool is built and its checksum self-test passes (2026-10-08); the full 8 GiB
> measurement has **not** been run. No DRAM bandwidth is measured yet, and nothing may quote one
> until `runs/glm53-flash/step04/dram-bw.json` exists with verdict `valid`.

## Method

| Part | Choice |
|---|---|
| Buffer | `VirtualAlloc`, >= 8 GiB (the 285K has 36 MB of cache; STREAM asks for 4x the last-level cache). Smaller is refused outside `--selftest`. Refused too when available RAM < buffer + 1 GiB. |
| Content | the u64 word index, `w[i] = i`, written by all cores before the first run |
| Kernel | AVX2: 4 aligned 256-bit loads per iteration into 4 `vpaddq` accumulators; reads only, no stores to the buffer |
| Work split | 4 MiB chunks claimed from one atomic counter, so fast and slow cores stay busy; `--passes` (default 8) passes over the buffer per run |
| Threads | one per physical core (SMT siblings are dropped), each pinned with `SetThreadGroupAffinity`; start released by a barrier; wall time = last thread end - first thread start |
| Core classes | `GetSystemCpuSetInformation`, grouped by `EfficiencyClass`; highest class = `p_cores`, lowest = `e_cores`, `all` = every core. Not hard-coded; a machine with one class gets `all` only |
| Statistic | one unmeasured warm-up run per configuration, then `--reps` (default 5, minimum 3) measured runs, interleaved across configurations; median, min, max, spread = max/min; GB/s = 10^9 B/s |

## Validity rules (fixed before any result)

| Rule | Outcome |
|---|---|
| sum of the words read != n(n-1)/2 x passes (mod 2^64) in any run, warm-up included | `invalid_checksum` |
| any run above 89.6 GB/s (`--nominal-gbs`) | `invalid_above_nominal`: that is cache or a wrong buffer, not DRAM |
| spread max/min > 1.15 | `void_spread` |
| otherwise | `valid` |

The checksum has a closed form, so it does not depend on a second read of the buffer. A read loop
that is removed or optimised away yields sum 0 and fails every run (shown once for #170).

## Run

Build (Windows x86_64 only; other targets exit 2):

```
cd probes
set CARGO_BUILD_JOBS=4
cargo build -p crow-nest-probes --release --bin p17_dram_bw
```

Self-test, 256 MiB, under a second, checks the checksum only (its GB/s are not a measurement):

```
target\release\p17_dram_bw.exe --selftest
```

Full measurement, **only on an otherwise idle machine and never next to a download, a serve or any
long job**, and only after robin's go:

```
cargo run -p crow-nest-probes --release --bin p17_dram_bw -- --gib 8 --reps 5 --out ..\runs\glm53-flash\step04\dram-bw.json
```

## Output

JSON on stdout (and `--out`): machine (cores, logical processors, cores per efficiency class,
available RAM before), buffer, passes, reps, and per configuration the rates of every run, median,
min, max, spread, `checksum_ok`, `verdict`; top-level `verdict` is `valid` only if all three are.
Progress goes to stderr. Exit 0 valid, 1 void/invalid/self-test failure, 2 refusal.

## Limits

- Read bandwidth with 8 GiB of 4 KiB pages: no large pages, so TLB misses are included. A figure
  with large pages could be higher; not measured.
- Pure reads, no writes, no latency, one thread per core. A CPU lane that also writes or runs with
  other threads active can see less; not measured.
- STREAM reports the best of ~10 trials; this tool reports the median with the spread, which gives
  a lower number than best-of and a statement about run-to-run stability.
