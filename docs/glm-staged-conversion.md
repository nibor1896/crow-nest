# Staged GLM-5.3-Flash conversion (download, convert, delete behind)

Step 9 of the GLM-5.3-Flash plan (#157): the full CNQ4.5 container from
`zai-org/GLM-5.3-Flash` rev `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a`, shard by shard, because C:
cannot hold the 62 originals (328,337,455,672 B) and the container (178,478,618,624 B without the
index trailer, `converter plan --headers`, `converter/README.md`) at once. PREREG:
`runs/glm53-flash/PREREG.md` (scale policy `--scales mse`).

## Three processes, one directory

All three work on `models/GLM-5.3-Flash-original/` and keep their state only in files, so each can
be killed and started again at any point.

| process | writes | reads | log |
|---|---|---|---|
| `tools/fetch-glm.py --shards 1-62 --wait-for-space` | `<shard>.part` → `<shard>` + `<shard>.verified` (size and sha256 == HF record) | `hf-revision.json`, `<shard>.deleted` (skip) | `fetch.log` |
| `converter … --headers … --consume …` | the container, `<out>.cnq.journal.jsonl`, `<out>.cnq.sidecar.jsonl`, `<shard>.done` | `<shard>.verified`, `<shard>`, `headers/` | its stderr |
| `tools/glm-stage.py --out <container>` | `<shard>.deleted`, then deletes `<shard>` | `.done`, `.verified`, journal, container, `headers/`, `config.json` | `stage.log` |

The converter writes in (shard, offset) order and every GLM write unit reads one shard
(`converter/README.md`), so it consumes shards 1 → 62 — the order the downloader fetches them in.
The downloader can therefore run ahead and park at the 20 GB reserve without a deadlock: the shard
the converter waits for is always the next one the downloader fetches, and everything before it
is deleted by then.

## Commands

From the crow-nest root (`C:/Users/robin/dev/crow-nest`); `D` = `models/GLM-5.3-Flash-original`,
`OUT` = `converter/GLM-5.3-Flash-CNQ4.5.cnq`. Use absolute paths for `OUT` in both commands: the
supervisor compares the `out` the converter writes into `.done` with its own `--out`.

| step | command |
|---|---|
| download (runs ahead, parks at 20 GB) | `python -I tools/fetch-glm.py --dest D --curl C:\Windows\System32\curl.exe --shards 1-62 --wait-for-space` |
| build the converter (needs `--consume`, crow-nest #155) | `cd converter && cargo build --release` |
| convert | `converter/target/release/converter.exe --scales mse --source-repo zai-org/GLM-5.3-Flash --revision eb9eb208eb0d988989d07a6a12d0fdeb5f52574a --headers D/headers --consume D D OUT` |
| delete behind | `python -I tools/glm-stage.py --dest D --out OUT` |
| status (no lock, no change) | `python -I tools/glm-stage.py --dest D --out OUT --status` |
| status per file | `python -I tools/fetch-glm.py --dest D --status` |
| what would be deleted | `python -I tools/glm-stage.py --dest D --out OUT --once --dry-run` |

The supervisor exits 0 once all 62 shards are deleted; the converter then writes the index trailer
and removes its journal. Start the supervisor together with the converter, never before the
shards it would delete are no longer needed by anything else (step 6 reads shards 1, 2, 3, 17,
31, 32, 62 from the same directory).

`--status` prints one line:

```
stage: verified 4/62, on disk 4, done 0/62, deleted 0/62 | shards on disk 21,413,506,520 B + .part 106,442,752 B | free 313,505,275,904 B (313.5 GB) | container 0 B (0.0 GB), no journal
```

`verified` counts `.verified` markers (they stay after a delete), `on disk` the shard files,
`done` the converter's markers, `deleted` the shards gone with a `.deleted` marker; then the bytes
of shard files and `.part` files, the free space of the volume, the container size and the
journalled tensor count (37,534 at the end, or `index trailer written`). The supervisor writes the
same line to `stage.log` at start, every 25 minutes (`--status-every 1500`, the owner's interval)
and at the end.

## When a shard is deleted

Every pass (`--poll`, 10 s) `glm-stage.py` deletes a shard only if all of these hold, and logs
`KEEP <shard>: .done present but <reason>` once per reason otherwise:

1. `<shard>.done` exists and its `out` is this container (relative paths are tried against the
   working directory and the repository root).
2. The shard is verified: its `.verified` record equals the HF record and the file has its size.
3. Every tensor the `cnq4.5-glm5-next` recipe writes from the shard is in the journal's verified
   prefix. The tensors come from `headers/<shard>.json`, minus what the recipe omits (vision tower,
   language layers ≥ `text_config.num_hidden_layers` = 45, the MTP block); an FP8
   `X.weight_scale_inv` counts through `X.weight`. Over the 62 real headers this rule requires
   37,534 names — the 37,534 tensors `converter plan` writes (2026-10-08). The verified prefix is
   the converter's own resume criterion: records in `seq` order, `offset == previous end + pad`,
   zero alignment bytes, and the sha256 of the container bytes equal to the record's. A resumed
   converter therefore never needs a deleted shard again. After the trailer is written (journal
   removed) the index trailer (`format_version 2`) is the record instead.

Then it writes `<shard>.deleted` (shard, size, sha256, out, tensor count, journal records, time)
and deletes the file: `DELETED <shard>: <bytes> B, sha256 …, N tensors journalled (journal M
records); free now … B`. A kill between marker and delete is completed on the next pass after the
same checks; a shard gone without a marker (deleted by hand after its `.done`) gets its marker
(`marked …`). The supervisor never deletes a `.part`, `.verified` or `.done` file, nor any file
that is not one of the 62 shards of the HF record. One supervisor per directory (`stage.lock`).

## Disk

- The downloader never starts or continues a transfer that would leave less than 20 GB free; in
  `--wait-for-space` mode it parks instead of stopping (`WAIT …`, `LOW SPACE …`, `space back …` in
  `fetch.log`).
- The 20 GB bind the downloader's own bytes. The converter writes a shard's container share
  (≈ 2.9 GB on average: 178.48 GB / 62, derived) before that shard's `.done`, and the shard goes
  up to one pass (10 s) later, so with the downloader parked the free space can dip that far below
  the parking level — derived, not measured.
- Run-ahead peak (derived): 313.5 GB free with 4 shards on disk (2026-10-08 02:03) leaves room for
  ≈ 55 more before the reserve, so the download parks at about shard 58–59 until the conversion
  starts; the job then holds ≈ 310 GB of shards. That is above the ≤ 207 GB peak the ticket
  expected for a 3-shard window; the window mode is not what runs here.
- Each converted shard frees ≈ 5.3 GB and adds ≈ 2.9 GB of container, so the free space grows
  while the converter catches up and the downloader resumes once a whole shard fits above 20 GB.

## Recovery

- Any process killed: start it again with the same command. The downloader resumes `.part` files
  and skips verified and `.deleted` shards; the converter resumes from its journal; the
  supervisor re-reads the journal from the start (it re-hashes the journalled container bytes
  once, ≈ the container size).
- A converter resume that needs a deleted shard (only if container bytes of the verified prefix
  changed on disk afterwards): it stops with "… (it has a .verified marker)". Remove that shard's
  `.deleted`, `.done` and `.verified` markers and run the downloader again.
- A `.done` of another conversion in the directory (e.g. a step-6 run with `--consume`) is never
  enough: `KEEP … .done names another container` until this conversion writes its own.

## Tests

`python -I tools/test_glm_stage.py` — the omission and scale rule, the journal's verified prefix
(corrupt byte, nonzero alignment, half-written line, rewritten journal, foreign file), the trailer,
and the delete pass on a small world in the converter's file format: deleted once and marked,
idempotent, never without `.done`, never for another container, never with a tensor missing or a
journalled tensor that does not re-hash, never for an unverified shard, completion after a kill,
dry run, the status line, and the downloader skipping what was deleted. No network, no converter.
2026-10-08: 25 tests, OK. The downloader's side: `python -I tools/test_fetch_glm.py` (37 tests, OK).
