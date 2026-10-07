# Downloading GLM-5.3-Flash (`tools/fetch-glm.py`)

The originals of `zai-org/GLM-5.3-Flash` at revision `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a`,
fetched from that source only and verified file by file. Used by the GLM-5.3-Flash series:
small files and shard headers for step 4 (#154), the shards of layers 0–3 for step 6 (#156), and
the download half of the staged full conversion in step 9 (#157). Data lands in
`models/GLM-5.3-Flash-original/` (git-ignored).

## Commands

Run from the repository root; `-I` because the tool reads downloaded JSON. Outside the main
checkout (a worktree) pass `--dest C:/Users/robin/dev/crow-nest/models/GLM-5.3-Flash-original`.

| command | does |
|---|---|
| `python -I tools/fetch-glm.py --small` | the 9 small files (28,715,382 B): `config.json`, `generation_config.json`, `model.safetensors.index.json`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, `processor_config.json`, `README.md`, `LICENSE` |
| `python -I tools/fetch-glm.py --headers` | the 62 shard headers by HTTP range → `headers/<shard>.json` (`{"shard", "size", "data_start", "header"}`) |
| `python -I tools/fetch-glm.py --shards 1,31-32` | whole shards by number (1-based) or file name |
| `python -I tools/fetch-glm.py --for-layers 0-3 --with-embed-head` | the shards holding any tensor of those language-model layers, plus `embed_tokens`, `lm_head` and the final norm, resolved from the verified index |
| `… --for-layers 0-3 --with-embed-head --plan` | print that shard set with sizes; fetch nothing |
| `python -I tools/fetch-glm.py --status` | one line per file: bytes on disk / expected, verified yes/no; header cache count; no network |

Every action is logged with a timestamp to `models/GLM-5.3-Flash-original/fetch.log`; a run ends
with one `run end:` line (bytes added, curl runs, range requests, curl-internal retries, outer
restarts, hash mismatches) — the raw material for the measurement-book row.

## Reference and verification

The reference is the HF API of the revision, `GET /api/models/zai-org/GLM-5.3-Flash/revision/<rev>?blobs=true`,
saved once as `hf-revision.json` (the revision is immutable; `--refresh-api` re-reads it). The tool
refuses an answer for another revision, a shard without `lfs.sha256`, or a shard count other than 62.

- **Shards and `tokenizer.json`** (LFS): size == `size` and sha256 == `lfs.sha256`.
- **Other small files** (plain git): size == `size` and the git blob id — SHA-1 over
  `"blob <size>\0" + bytes` — == `blobId`. The sha256 is logged too.
- A file is downloaded as `<name>.part`; only after both checks pass is it renamed to `<name>` and
  a marker `<name>.verified` (size, sha256, blob id, repo, revision, time) written. `--status`
  and later runs trust the marker plus the size, without rehashing. A final file without a
  marker is rehashed before it is trusted.
- **Mismatch → delete, fetch again, never keep.** After 3 mismatches on one file the run stops.
- **Headers:** each `headers/<shard>.json` is checked to tile its shard exactly (tensor byte
  ranges contiguous from 0, `data_start + Σ tensor bytes == size`). Measured 2026-10-08:
  62/62 shards, 76,108 tensors, 10,684,096 B of headers (incl. the 8-byte length fields) +
  328,326,771,576 B of tensors = 328,337,455,672 B = the shard total; 62 range requests, 0 retries.

## Retry, stall and disk rules

Hugging Face drops long transfers (TLS resets, WinError 10054, curl exit 35). Two layers:

1. **curl:** `-L -f -C - --retry 20 --retry-all-errors --retry-delay 10 --connect-timeout 30
   --speed-limit 262144 --speed-time 120` (a transfer slower than 256 KiB/s for 120 s aborts and
   is retried).
2. **Outer loop:** restarts curl with `-C -` (resume from the bytes on disk) with backoff
   10 s → 300 s (reset after progress) until the byte count equals the API size; a size above
   it deletes the `.part`. A watchdog kills a curl whose file has not grown for 600 s.
   HTTP 401/403/404/410 stop the file at once (wrong repo, revision or access).

Known limitation: curl's own `--retry` does **not** resume — it truncates the file back to where
that curl run started (curl `docs/TODO.md`, "--retry should resume"). A drop inside one curl run
therefore costs the bytes of that run (≤ one shard, ≈ 5.4 GB); only the outer loop resumes.

**Disk:** before each shard the free space of the destination volume is checked; a shard that
would leave less than 20 GB free is refused and the run stops (exit 4). One run per destination:
`fetch.lock` with a heartbeat (stale after 120 s); `--status` works while a run holds it.

Exit codes: 0 ok, 3 a file failed (permanent HTTP error, 3 mismatches, header sum off), 4 refused
for disk space, 130 interrupted.

## Step 6 shard set (#156)

`--for-layers 0-3 --with-embed-head --plan` on the verified index (2026-10-08):

| shard | bytes | holds |
|---|---|---|
| `model-00001-of-00062` | 5,365,306,704 | `embed_tokens`, `lm_head` |
| `model-00002-of-00062` | 5,320,647,824 | layers 0, 1 |
| `model-00003-of-00062` | 5,363,467,432 | layer 1 |
| `model-00017-of-00062` | 5,364,084,560 | layer 2 |
| `model-00031-of-00062` | 5,364,341,544 | layer 3 |
| `model-00032-of-00062` | 5,363,915,232 | layer 3 |
| `model-00062-of-00062` | 1,261,584,968 | final norm `model.language_model.norm.weight` |
| **7 shards** | **33,403,348,264** | tensors: layer 0–2 29 each, layer 3 1,762, embed/head/norm 1 each |

## Tests

`python -I tools/test_fetch_glm.py` — the pure parts and the outer loop against a fake curl
(resume to the exact size, mismatch deleted and refetched, 3 mismatches stop, overlong deleted,
permanent HTTP error stops, 20 GB floor refuses, an unmarked file is rehashed). No network.
