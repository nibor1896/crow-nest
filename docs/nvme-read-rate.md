# NVMe read rate B for expert-sized blocks

`tools/nvme_read_rate.py` is step 3 of the GLM-5.3-Flash series (issue #146, parent
nibor1896/Crow#362). It measures one number the plan cannot go on without: the sustained rate B
at which this machine's NVMe delivers expert-sized blocks through unbuffered IO. The method and
the verdict are fixed by `runs/glm53-flash/PREREG.md`, section "Step 3"; G1 (step 8) consumes B.
The tool changes nothing in the engine, the converter or the container.

> **No B exists yet.** The tool is built and tested on a synthetic temp file (2026-10-08). The
> measurement on the GLM shards has not been run: the shards were still downloading, and a run
> with a second process on the disk is void.

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

## Not covered

- The ticket's sketch also named a cache-blindness control (buffered pass, then a direct pass
  within 5 % of the flushed rate) and a sequential floor of 3,000 MB/s. PREREG step 3 sets
  neither; the tool proves the unbuffered flag by the offset-1 refusal instead, and reports the
  sequential rate without a threshold.
- The `--split` arm (one visit as 9,437,184 B + 4,718,592 B at two offsets) is not built.
- This is a Python harness over `ReadFile`, not the NVMe tier backend of step 14; whether that
  backend reaches the same B is step 14's question.
- Linux (`O_DIRECT`) is not implemented.
