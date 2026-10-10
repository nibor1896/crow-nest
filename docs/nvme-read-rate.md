# NVMe read rate B for expert-sized blocks

`tools/nvme_read_rate.py` is step 3 of the GLM-5.3-Flash series (issue #146, parent
nibor1896/Crow#362). It measures one number the plan cannot go on without: the sustained rate B
at which this machine's NVMe delivers expert-sized blocks through unbuffered IO. The method and
the verdict are fixed by `runs/glm53-flash/PREREG.md`, section "Step 3"; G1 (step 8) consumes B.
The tool changes nothing in the engine, the converter or the container.

Issue #171 (plan step 6, child of #169) adds a second, optional arm: B per queue depth and reader
count for the two record sizes, with latency per block and the drive temperature. It is off unless
`--depths` is given and is described in "Queue-depth grid" below; the PREREG arm and its verdict are unchanged.

> **First run on the GLM shards (2026-10-08, `runs/glm53-flash/step03/20261008T001819Z.{json,md}`):** valid
> (no foreign disk IO), medians 6.994 / 9.765 / 7.811 GB/s at 1 / 2 / 4 readers, spreads 1.003 / 1.232 / 1.430.
> The best count (2) misses the 1.15 spread rule, so `G1 input` is "not answered". PREREG amendment 5 (robin, 2026-10-08)
> sets B for G1 = 6.994 GB/s, this run's 1-reader median; the binding reader count for step 14 is 1.

## What it reads, and how

| Arm | What | PREREG |
|---|---|---|
| random | blocks of 14,155,776 B (one expert: 3 × 4096 × 2048 weights, 36 B per 64 values) at 4096-aligned uniform random offsets over all given files; `FILE_FLAG_NO_BUFFERING`; one handle per reader and file; one `VirtualAlloc` buffer per reader; 1, 2, 4 readers; ≥ 3 repetitions, interleaved (1, 2, 4, then again) | Step 3 method, n, statistic |
| sequential | one reader, unbuffered, the files front to back in 16 MiB reads, 3 passes | "reported beside it, no threshold" |

A cell runs `--warmup` unmeasured seconds (default 2), then `--secs` measured seconds (default
20). A read counts when it returns the whole block and completes inside the measured window;
the rate is counted bytes / `--secs`, in GB/s (10⁹ B/s). Each cell also reports the p50 and p99
time per block.

Before the first read the tool refuses (exit 2) when the block or the alignment is not a
multiple of the volume's logical and physical sector size (`IOCTL_STORAGE_QUERY_PROPERTY`,
`StorageAccessAlignmentProperty`, no fallback), when a file is smaller than one block, when the
files lie on more than one physical disk, or when a read at offset 1 succeeds (the unbuffered
flag did not take; it must fail with `ERROR_INVALID_PARAMETER`).

## Output

Per cell: repetition, reader count, GB/s, blocks, p50/p99 ms, short reads, foreign read/write
MB/s. Then the verdict per reader count:

- every repetition's rate, the median (= B for that reader count), min, max;
- the spread max/min and whether it is ≤ 1.15;
- m* = B / 190.3 GB/s, the largest share of expert visits that may come from the NVMe at
  40 tok/s (190.3 GB/s = 40 tok/s × 4.7563 GB per token if every visit came from disk);
- the prefill bound 39.9 × B tok/s and whether it reaches the G5 cold-prefill line of 150 tok/s
  (judged in G1, printed here for reading);
- the sequential median beside it;
- `G1 input`: B = the median at the reader count with the best median, which is binding for
  step 14, but only if the run is not void, reader counts 1, 2 and 4 were measured, there were
  ≥ 3 repetitions, and that count's spread is ≤ 1.15. Otherwise "not answered" or "VOID" with
  the reason. If the best count's spread fails, the tool does not fall back to another count.

`--json PATH` writes every cell, the environment (UTC date, crow-nest commit, drive model and
firmware, physical disk, volume, free bytes, sector sizes, all arguments, file sizes) and the
verdict. Run directories belong under `runs/glm53-flash/step03/` (ignored by git).

## Queue-depth grid (#171)

```
python -I tools/nvme_read_rate.py --block 3.05bit --depths 1,4,8,16 --readers 1,2,4 --reps 3 --secs 20 \
    --backend iocp --no-sync-arm --seq-passes 0 --json runs/glm53-flash/c1-s6-nvme/<UTC stamp>-3.05bit.json <shard> [<shard> ...]
```

| Flag | Meaning |
|---|---|
| `--block` | bytes per read, or a preset: `3.05bit` = 9,474,048 B (2313 × 4096), `4.5bit` = 14,155,776 B (3456 × 4096, the default). Offsets and buffers are 4096-aligned; the sector-multiple check applies as before. The 9,474,048 B figure is one GLM-5.3-Flash expert record in the MUL1 trellis codec at K = 3 (the expert bitrate of the 3.05 bpw EXL3 checkpoint): 3 × 3,145,728 B trellis + 36,864 B fp16 suh/svh, no padding (`converter/src/mul1.rs` `glm_record_size_at_3_bit`, #181). |
| `--depths` | queue depths per reader, distinct positive integers, e.g. `1,4,8,16`. Turns the grid on. |
| `--readers` | reader counts of the grid (and of the PREREG arm), default `1,2,4`. |
| `--backend` | `iocp` (default): one completion port per reader, `GetQueuedCompletionStatusEx`. `ioring`: Windows 11 `CreateIoRing` / `BuildIoRingReadFile` / `SubmitIoRing` / `PopIoRingCompletion` over ctypes; refuses with exit 2 ("IoRing unavailable") if the system has none. |
| `--no-sync-arm` | skip the PREREG synchronous arm (grid only). `G1 input` then says "not measured". Needs `--depths`. |

Per cell, `readers` threads each own their handles (one per file, opened `FILE_FLAG_NO_BUFFERING |
FILE_FLAG_OVERLAPPED`, sharing read only) and `depth` `VirtualAlloc` buffers, and keep `depth` random
reads in flight at uniform 4096-aligned offsets until the deadline, then drain. The window, the
counting rule (a block counts when it completes whole inside the measured window) and the offset
sampler are the random arm's. The latency of a block is the time from its submit to the moment the
reader reaps it (a Python reader, so it includes the reaper's scheduling). At depth 1 the cell is the
random arm's synchronous case through the overlapped path. Memory: `readers × depth × block` bytes of
buffers (4 × 16 × 14,155,776 B = 0.9 GB at the largest cell).

Output per cell: repetition, depth, readers, GB/s, blocks, p50 / p99 ms, short reads, foreign r/w MB/s.
Then a matrix B(depth, readers) of medians, and per (depth, readers): median, min, max, spread max/min with
"≤ 1.15?", p50 (median of the repetitions' p50) and p99 (the worst repetition's p99) per block. `--json`
adds a `grid` object (`backend`, `block`, `depths`, `readers`, every `cells` entry, `summary`,
`temperature.before` / `.after`). Repetitions are interleaved (depth-major, then readers, then again).
The grid claims no `G1 input`; whether a cell's spread is ≤ 1.15 is printed, not enforced.

Drive temperature is read before the first and after the last grid cell with `IOCTL_STORAGE_QUERY_PROPERTY`
(`StorageDeviceTemperatureProperty`), hottest sensor in °C with the drive's warning and critical limits;
if the query fails the text is "not readable (<error>)". On this machine it is readable without admin
rights (smoke run 2026-10-08: 68 C before, 76 C after a 10 s run on a 200 MiB temp file, warning 87 C,
critical 89 C; a functional check, not a measurement row).

Void: the rule of the next section is applied to every grid cell (foreign reads, any writes, download
signs, a changing directory, a changing measured file, a file open for writing); the reasons are named
`grid rep R depth D readers N: ...`.

## Void: a second process on the disk

PREREG: "a run with a second process reading or writing the disk" is void. The tool checks it
with the physical disk's own counters (`IOCTL_DISK_PERFORMANCE` on `\\.\PhysicalDriveN`,
readable without admin rights on this machine):

- an idle check (`--idle-check`, default 3 s) before the first read, then every cell and every
  sequential pass: disk bytes read minus the tool's own bytes, and disk bytes written, per
  second of that window. Either above `--foreign-max-mbps` (default 4 MB/s) voids the run.
  Read on 2026-10-08 with these counters: background writes 0.24 and 1.24 MB/s (1 s and 2 s
  samples, ~01:30 CEST, before the GLM download started); the GLM shard download (`fetch.log`
  start 01:33:54) wrote in bursts, 3.1–32.5 MB/s in 1 s windows, 11.7 MB/s over 5 s, 23.8 MB/s
  over 10 s. A 1 s window can therefore miss a bursty writer; a 22 s cell does not, and the
  directory check below does not depend on the rate;
- a directory that holds a measured file and contains `*.part`, `*.partial`, `*.incomplete`,
  `*.tmp` or `fetch.lock` (a download in progress, or one left unfinished) voids the run before
  the first read; a stale lock has to be removed by hand after checking that no fetch runs;
- any entry of such a directory that appears, disappears or changes size or mtime during the run
  voids it (a growing `.part`, a finished shard, a new lock);
- a measured file whose size or mtime changed during the run voids it;
- every file is opened sharing read only, so a file another process holds open for writing
  refuses the open (exit 2: "open for writing by another process");
- `--disk-busy` tells the tool that another process uses the disk.

Void known at the start (`--disk-busy`, idle check) stops the tool before the first read with
exit 3. `--allow-void` runs anyway for functional tests; every number is then labelled VOID, `G1
input` is "VOID: …", and the exit code is 3. A void run never prints a valid verdict. The
counters see the whole physical disk, so they cannot tell whose IO it was; any foreign rate
above the limit counts, including a short burst from Windows itself, which then costs a re-run.

## Running it

```
python -I tools/nvme_read_rate.py --json runs/glm53-flash/step03/<UTC stamp>.json <shard> [<shard> ...]
```

Defaults are the PREREG values (block 14,155,776, align 4096, readers 1,2,4, 3 reps, 20 s per
cell, 3 sequential passes). Nine cells of 22 s plus three sequential passes: about 3.5 minutes
for 16–32 GB of shards. Readers 8 and 16 (where Crow#42 saw the knee) can be added with
`--readers 1,2,4,8,16`; they then also compete for "best reader count". Exit codes: 0 valid run,
2 refused, 3 void. Windows only; elsewhere it refuses with exit 2.

## Tests

```
python -I tools/test_nvme_read_rate.py
```

Pure parts on every platform: sector rules, the offset sampler (aligned, in bounds, every slot
reachable, files weighted by slot count), the verdict (spread boundary 1.15, B only at the best
count, no B when void or not a PREREG row), m* and the prefill bound, the foreign-IO arithmetic.
On Windows, end to end on a synthetic temp file that is deleted afterwards (rates from those
small cells mean nothing and are no measurement row), plus the void paths: `--disk-busy` with and
without `--allow-void`, a writer thread doing `fsync` beside the run, a `.part` file beside the
shard, a file appearing beside the shard during the run, a file held open for writing, the
unbuffered self-check, and the refusals.

#171 adds, on every platform: the block presets and `--depths` parsing; the in-flight loop (`run_queue`)
against a fake FIFO backend and a fake clock (exactly `depth` reads outstanding, no submit at or after
the deadline, drain, only completions inside the window count, short reads, latency submit to reap,
backend errors propagate); the B(depth, readers) summary, matrix and printout; the grid-only verdict;
and the temperature parser. On Windows, a 32 MiB synthetic file runs the grid end to end with `iocp`
and `ioring` (the latter skips itself when the system refuses with "IoRing unavailable"), with the
block preset, with the void label, and without `--depths` (no `grid` key).

## Not covered

- The ticket's sketch also named a cache-blindness control (buffered pass, then a direct pass
  within 5 % of the flushed rate) and a sequential floor of 3,000 MB/s. PREREG step 3 sets
  neither; the tool proves the unbuffered flag by the offset-1 refusal instead, and reports the
  sequential rate without a threshold.
- The `--split` arm (one visit as 9,437,184 B + 4,718,592 B at two offsets) is not built.
- This is a Python harness over `ReadFile`, not the NVMe tier backend of step 14; whether that
  backend reaches the same B is step 14's question.
- Linux (`O_DIRECT`) is not implemented.
- The queue-depth grid is not yet measured on the GLM shards (#171 builds the tool; the 328 GB download
  on C: must be finished and robin must give the go first). No B(depth, readers) number exists yet.
- The grid reaps in Python threads: its latency includes the reaper's scheduling, and it is a harness
  figure, not the NVMe tier backend's of step 14.
