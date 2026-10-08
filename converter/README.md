# crow-nest converter

| item | value |
|---|---|
| what it does | streams the original safetensors shard by shard and writes one CNQ container with an index v2 (Crow #300 C6, 2026-09-26; v1 before) |
| quantization | NVFP4 per ggml geometry, round to nearest, calibration free |
| what it never does | hold the model in RAM, judge its own output, or read a GGUF |
| spec | `../docs/architecture.md` section 1, approved 2026-09-02 |
| platform | Linux and Windows; the crate is pure Rust with no CUDA and no platform code, and it was untouched by the Linux port of 2026-09-17 (issue #15) |
| module comment of record | `src/main.rs:1-65` |

## Usage

```
usage: converter [--scales ceil|mse] --source-repo <org/name> [--revision <sha>] <model-dir | file.safetensors> <out.cnq>
  writes an index v2 container: config.json + generation_config.json verbatim, the family's recipe, source repo/revision/shard sha256
  (--revision defaults to the Hugging Face cache in the model dir; Crow #300 C6)
  --scales ceil  ceiling sub-block scales: stored >= raw always, max_rel <= 1.0 (default)
  --scales mse   per-sub-block SSE-minimizing scales: clipping allowed, quality via MSE report
  --scales diag --diag-stats <f.json>  all 126 ue4m3 steps scored by the activation-weighted error (Crow #300 p2-lh)
  --headers <dir>  read the shard headers from a header cache (<dir>/<shard>.json) (crow-nest #154)
  --consume <shard-dir>  convert while shards come and go: wait for <shard>.verified, write <shard>.done, never delete; needs --headers (crow-nest #155)
  an interrupted conversion resumes from <out>.cnq.journal.jsonl (crow-nest #155)
  --layers <spec> [--with-embed-head]  a partial container: text layers <spec> only (0-3, 0,3) [+ token embedding, lm_head, final norm]; the rest is filtered and the index says so (crow-nest #156)
  --experts-mul1 <store> [--mul1-wait] [--disk-reserve-gib N]  GLM-5.3-Flash: the routed experts (MTP layer 45 incl.) as MUL1 K=3 trellis records from a tools/glm_mul1_quantize.py store, after the dense part; --mul1-wait waits for records still being quantized; refused when free disk < bytes to write + N GiB (default 16) (crow-nest #182)
       converter [--scales ceil|mse] requant-check <dense.safetensors> <container.cnq>
  re-quantizes fetched originals and compares them with the container's own bytes (#76)
       converter dequant <container.cnq> (<name>[:<r0>:<r1>] ... | --names -)
  writes the named tensors (rows r0..r1) to stdout as f32 little endian, decoded as gate 0 decodes them (crow-nest #156)
       converter plan [--headers <dir>] [--source-repo <org/name>] [--revision <sha>] <model-dir | file.safetensors>
  the dry run: family, recipe, per-tensor dtype/section table, GPU / host byte totals (Crow #300 C6)
```

- The usage text above is the `HELP` constant verbatim (`src/main.rs:653`), without the overlay and `imatrix-show` lines.
- Input is a directory with `model.safetensors.index.json`, or a single `.safetensors` file.
- Two positional arguments are required; anything else exits 2.
- `config.json` and `generation_config.json` must sit in the model directory (or beside the single file): the index v2 carries them, and `text_config.model_type` picks the recipe. `--source-repo` is required; a missing one exits 2 before anything is written.
- `requant-check` is the read-only subcommand of issue #76; it is described below.

## Container layout

| range | content | source |
|---|---|---|
| `[0..4)` | magic `CNQ1` | `src/main.rs:53`, spec section 1 approved 2026-09-02 |
| `[4..12)` | reserved, zeros | `src/main.rs:54`, spec section 1 approved 2026-09-02 |
| `[12..)` | payload blob, streamed | `src/main.rs:55`, spec section 1 approved 2026-09-02 |
| `[end-8-index_len .. end-8)` | index JSON, UTF-8 | `src/main.rs:56`, spec section 1 approved 2026-09-02 |
| `[end-8 .. end)` | `u64` little endian `index_len` | `src/main.rs:57`, spec section 1 approved 2026-09-02 |

- The index is a TRAILER, not a header: the payload streams to disk without knowing the index size up front (`src/main.rs:5-7`).
- The old index-first layout needed the whole blob in RAM (`src/main.rs:6-7`).
- Tensor offsets in the index are relative to blob start, which is `12` (`src/main.rs:58`).

## NVFP4 geometry

| quantity | value | source |
|---|---|---|
| values per block | 64 | `src/main.rs:9`, spec section 1 approved 2026-09-02 |
| sub-block scales per block | 4, `ue4m3`, one per 16-wide k-block | `src/main.rs:9-10`, spec section 1 approved 2026-09-02 |
| packed `E2M1` payload per block | 32 B | `src/main.rs:10` |
| stored bytes per block | 36 B | `src/main.rs:10` |
| effective width | 4.5 bpw | `src/main.rs:10` |
| second level | one global `f32` scale per tensor | `src/main.rs:11` |
| calibration | none, round to nearest | `src/main.rs:11` |

## Index v2 and the per-model recipe (Crow #300 C6, 2026-09-26)

| item | value |
|---|---|
| index keys added | `format_version: 2` (replaces `version: 1`), `recipe`, `scales`, `model` |
| `model` block | `config_json` and `generation_config_json` verbatim as strings, each with its hash in `<name>_sha256`; `family`, `model_type`, `geo`; `source.repo`, `source.revision`, `source.shards[]` (`file`, `size`, `sha256`, `sha256_from`: `hf-lfs` or `computed`) |
| recipe rows | `cnq4.5-flash-next` (`qwen4_exp_text`): the keep set below, verbatim. `cnq4.5-qwen35-dense` (`qwen3_5_text`): the phase 2 recipe, a whitelist. `cnq4.5-glm5-next` (`glm5_next_text`): GLM-5.3-Flash, a whitelist, see below (crow-nest #154/#155) |
| new dtype | `f32`: the dense row's `A_log`, widened exactly from BF16 |
| dry run | `converter plan <model-dir>`: headers and configs only |
| source | `src/recipe.rs`, `src/main.rs` `index_v2`; full table in `../docs/architecture.md` 1.7 |

- The Flash-Next row is proved against all 1658 tensors of the CNQ4.5-M index (`tests/fixtures/cnq45m-index.tsv`, extracted 2026-09-26).
- The engine accepts a v1 index only for the CNQ4.5-M container of record (`../engine/src/cnq.rs`, `CNQ45M_INDEX_SHA256`).

## GLM-5.3-Flash: FP8 input, the `cnq4.5-glm5-next` row, the staged conversion (crow-nest #154, #155, 2026-10-08)

Source (read 2026-10-08): `zai-org/GLM-5.3-Flash` rev `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a`, 62 shards, 76,108 tensors. Plan of record and PREREG: `../runs/glm53-flash/PREREG.md`.

| item | value | source |
|---|---|---|
| FP8 input | `F8_E4M3` (`e4m3fn`: bias 7, no infinities, 0x7F/0xFF NaN and refused) with one F32 `weight_scale_inv` per 128x128 tile, row-major grid `[ceil(r/128), ceil(c/128)]`, partial edge tiles allowed; f32 value = `lut[q] * scale`, one f32 multiply, as DeepSeek-V3 `weight_dequant` and transformers `Fp8Dequantize` (2026-10-08) | `src/fp8.rs` |
| pairing | every FP8 `X.weight` needs `X.weight_scale_inv` with the grid shape and dtype F32/BF16; a weight without scale, a scale without weight, a non-2-D FP8 tensor: refused by name. Scales are read, never written (2026-10-08) | `src/main.rs` `build_manifest_from` |
| unknown dtype | refused by name (was: skipped with a line on stderr) | `src/main.rs` `build_manifest_from` |
| config check | `num_hidden_layers` 45 + `num_nextn_predict_layers` 1, `quantization_config` fp8 / e4m3 / [128, 128], or refused (2026-10-08) | `src/recipe.rs` `check_family_config` |
| omitted | vision tower (`model.visual.*`, 347 tensors) and the MTP block (layer 45, 1,760 tensors incl. scales): not converted in v1 (plan), named in the plan's totals (2026-10-08) | `src/recipe.rs` `omitted` |
| keep BF16 | token embedding (host RAM), `lm_head`, router `mlp.gate.weight`, norms, mHC `hc_*_fn`, DSA indexer, KDA gates `f_a/f_b/g_a/g_b/b_proj`, any 1-D, anything not a whole 64-value block (2026-10-08) | `src/recipe.rs` `decide_glm5_next` |
| carry F32 | `e_score_correction_bias`, KDA `A_log` / `dt_bias`, `hc_*_base` / `hc_*_scale` (F32 on disk) (2026-10-08) | same |
| NVFP4 | routed experts, shared expert, dense MLP (layers 0-2), MLA `q_a/q_b/kv_a/o_proj` (FP8 source) and `kv_b` (BF16 source), KDA `q/k/v/o_proj` and `q/k/v_conv1d` (BF16 source); each its own named row | same |
| anything else | refused by name: the row is a whitelist | same |
| write layout | the gate, up and down projection of one routed expert of one layer are one write unit, back to back, starting at an absolute file offset that is a multiple of 4096 (zeros before it, counted). Every other family: unchanged order, no zeros (2026-10-08) | `src/main.rs` `write_units` |

`converter plan --headers <dir>` on the 62 real headers (2026-10-08, branch `glm-converter`): 37,534 tensors written, 36,467 FP8 pairs (0 across two shards), dense resident part 5,981,546,744 B (5.982 GB; text without routed experts and the token embedding), routed experts 171,228,266,496 B, container without index trailer 178,478,618,624 B (178.479 GB, alignment zeros 28,412 B); every write unit reads one shard. The tensor table is `tests/fixtures/glm53-flash-tensors.tsv` (experts collapsed to ranges; `tests/fixtures/glm53_tensor_table.py` writes it from the header cache).

### Code histograms and the static coder (PREREG gate G2)

- Every NVFP4 sidecar line carries `class`, `layer`, `expert`, `h_codes` (16 counts: the nibbles as written, sign included) and `h_scales` (256 counts: the ue4m3 bytes as written). They are counted from the written 36-byte blocks, so `sum(h_codes) == n` and `sum(h_scales) == n / 16` also where `--scales mse` clips (2026-10-08).
- Classes (GLM): `expert_gate`, `expert_up`, `expert_down`, `shared_expert`, `attn_mla`, `attn_kda`, `dense_mlp`, `indexer`, `rest`. Other families: `expert`, `shared_expert`, `attn`, `attn_linear`, `dense_mlp`, `rest`.
- After the payload the sidecar gets `record: "code_summary"` lines: `scope` `class` (one table per class), `layer_class`, `tensor_tables` (one table per tensor, summed), `expert_blocks` (one table per routed expert block = the G2 metric, with min/max block saving) and `expert_blocks_layer`. Each has `raw_bytes`, `coded_bytes`, `table_bytes`, `entropy_bytes`, `saving`, `saving_entropy`.
- `coded_bytes` is exact: canonical Huffman by package-merge, length-limited to 12 bits (codes) and 15 bits (scales), `ceil(sum count * length / 8)` per stream, plus the code-length tables (8 B + 128 B). `entropy_bytes` is the order-0 bound. Source `src/entropy.rs` (2026-10-08).

### Staged conversion: `--headers`, `--consume`, resume

- **Missing shards** (the question of step 4): without `--headers`, the manifest opens every shard named in the index for its header and refuses a missing one; a conversion also hashes every shard without an LFS record and opens each shard at write time. So the plain mode needs all shards on disk.
- `--headers <dir>`: the headers come from `<dir>/<shard>.json` (`{"shard", "size", "data_start", "header"}`); `plan` then needs no shard. Provenance comes from `hf-revision.json` beside the configs (HF model info: `sha`, `siblings[].lfs.sha256/size`) as `sha256_from: hf-lfs`, without reading a shard; a shard that is on disk must have the recorded size (2026-10-08).
- `--consume <shard-dir>` (needs `--headers`): tensors in index order; before reading a shard the converter waits for `<shard>.verified` (written by the downloader after size and sha256), checks the shard's size and the marker's `sha256` against the HF record, and refuses on a mismatch. When every tensor that reads a shard is written and synced it writes `<shard>.done`. Deleting shards is the driver's job, never the converter's. On GLM-5.3-Flash every write unit reads exactly one shard, so the converter needs one shard at a time; the driver keeps a shard until its `.done` (2026-10-08).
- **Resume**: every conversion keeps `<out>.cnq.journal.jsonl`: line 1 names the recipe, the scale policy and the sha256 of the write order; then one record per tensor, appended after the container bytes are synced, and synced itself. Run the same command again after a kill: records whose bytes re-hash correctly are kept (in order, up to the first that does not), the container is truncated behind the last one, the sidecar is rewritten from them, and the run goes on. The journal is removed after the index trailer is written; a container without trailer is invalid. A journal of another plan (other scale policy, other tensors) is refused. A resumed run writes the same container and sidecar bytes as an uninterrupted one (test `a_glm_conversion_killed_and_resumed_is_byte_identical`); for that `serde_json` is built with `float_roundtrip` (2026-10-08).

### Partial container and `dequant` (crow-nest #156, plan step 6)

- `--layers <spec> [--with-embed-head]` (`src/partial.rs`): keeps the text decoder layers of `<spec>` (`0-3`, `0,3`, `0-2,7`) and, with the second flag, `embed_tokens`, `lm_head` and the final `norm`. The manifest is built over every tensor first (whitelist, FP8 pairing, config-vs-weights geometry), then filtered. A layer in `<spec>` without a written tensor is refused (exit 2).
- The coverage check counts the dropped names (weights and their block scales) as filtered, not missing: `coverage check (PARTIAL container, --layers 0-3 --with-embed-head): … filtered by the flags, … 0 missing`. The index trailer gets `"partial": {"layers", "embed_head_norm", "filter", "tensors_written", "tensors_filtered"}`; the journal head names the filter.
- `converter dequant <container.cnq> (<name>[:<r0>:<r1>] ... | --names -)` (`src/dequant.rs`, 2026-10-08): writes the named tensors, or rows of a 2-D tensor, to stdout as f32 little endian, no header, in the order asked. NVFP4 goes through `dequant_nvfp4`, which uses `nvfp4_scale` / `nvfp4_value`, the two functions gate 0 (`quantize_nvfp4_w`) decodes the written encoding with; BF16 is widened exactly, F32 read as stored. The oracle's `--weights container` back end reads every weight through it (`../docs/glm5-reference-runner.md` section 3).
- Step 6 run (2026-10-08, `--scales mse --headers … --layers 0-3 --with-embed-head` over the 7 step-6 shards): 972 tensors (902 NVFP4, 70 BF16/F32), payload 7.22 GB, 244 s, coverage 0 missing; numbers in `../runs/glm53-flash/step06/`.

### MUL1 K=3 experts from a quantizer store (crow-nest #182, plan steps 11/12, 2026-10-09)

- `--experts-mul1 <store>` (`src/mul1_store.rs`, `src/main.rs` `mul1_*`, 2026-10-09): GLM only. Every routed expert, the MTP block's 288 included (section `mtp`; the rest of layer 45 stays omitted), is one MUL1 K=3 record of `mul1::RecordLayout::size` B (9,474,048 B for GLM-5.3-Flash) built by `mul1::write_record` from the store's `L<ll>/E<eee>.safetensors` after its sha256 matched the store journal. The FP8 expert weights are not read here; `tools/glm_mul1_quantize.py quantize` read them.
- Write order: the dense part first, in the `cnq4.5-glm5-next` units and with its bytes, then one 4096-aligned record per expert in (layer, expert) order. Index entries: dtype `mul1` at the three trellis starts, `len` T / T / size - 2T, `mul1: {k, record_offset, record_bytes}`; top-level `expert_codec` block. The engine reads the codec and record size from that (`../engine/src/nvme_source.rs` `glm5_record_from_index`) (2026-10-09).
- `--mul1-wait` writes each record as the quantizer journals it; without it a missing record is refused before a byte is written. After a record's last projection is journalled the converter writes `<store>/L<ll>/E<eee>.done` (`{"out": <container>}`); the quantizer's `--prune-consumed <container>` deletes those record files. The converter deletes nothing.
- Disk check before the output is created or grown: the container bytes still to write plus `--disk-reserve-gib` (default 16) against the free space of the output's volume, else exit 2 with the numbers. Refused too: `--consume` beside it, a store of other shapes, K or record size, a changed record file, a tensor of another dtype or shape (2026-10-09).
- Resume as every conversion: the journal head names the store (`store.json` sha256), so a journal of another store is refused; a resumed run writes the uninterrupted bytes. `--layers` works as in #156 (2026-10-09).
- `converter dequant` does not decode `mul1` (refused by name); the Hadamard un-rotation belongs to plan step 10/11. The whole pipeline and its commands: `../docs/glm-mul1-conversion.md` (2026-10-09).

## BF16 keep set (the Flash-Next row)

- Approved as spec 1.2 on 2026-09-02, source `src/main.rs:13-17`.

- Embeddings and `lm_head`.
- The router GEMM.
- `shared_expert_gate`.
- All norms.
- Every 1-D tensor: biases, `A_log`, `dt_bias`, gates. Reason: under 0.1 % of the bytes, and a 1-D recurrence parameter must not go through a GEMM quant path.
- Any tensor whose length is not a multiple of `64` (`src/main.rs:17`).

## Section marks

- Decided 2026-09-02, source `src/main.rs:19-21`.

| mark | content | load |
|---|---|---|
| `text` | the default section | always |
| `ple` | `ngram_embedding`, NVFP4, block exchangeable to FP8 | always |
| `vit` | `model.visual`, carried in the Flash-Next container; the dense 27B recipe does not write it (the 27B uses `mmproj-F16.gguf`) | optional |
| `mtp` | carried in the container | optional |

## Verification sidecar

- Source `src/main.rs:23-26`.

- File name: `<out>.cnq.sidecar.jsonl`.
- One JSON line per tensor with max and mean error against the sub-block scales.
- Computed during quantization by dequantizing in place, so it costs no second pass.
- It is gate 0 of the measurement ladder.
- Exit code 1 on any bound violation, in the mode that has a bound.
- The converter never judges its own output beyond that bound; FP8-KV and PLE-NVFP4 quality are oracle comparisons (`../docs/architecture.md:119-120`, and section 5 for the gates).

## Scale policy `--scales`

| policy | sub-block scale | bound | exit gate | source |
|---|---|---|---|---|
| `ceil` (default) | smallest `ue4m3` ladder step at or above the sub-block max | stored is always at or above raw, so values never clamp; per-element error is bounded by one `E2M1` half-gap; `max_rel <= 1.0` | exit 1 on any violation | `src/main.rs:29-31`, spec section 1 approved 2026-09-02 |
| `mse` | the `ue4m3` ladder step minimizing the summed squared error of the 16 `e2m1` values, by analytic pre-selection plus local refinement | clipping is DELIBERATE: elements above six times the scale are cut, so `max_rel <= 1.0` does NOT hold and no fixed relative bound holds | disabled; quality is reported instead | `src/main.rs:32-41`, spec section 1 approved 2026-09-02 |

- The `mse` mode is the one that voids the bound.
- Read that row before quoting a `max_rel` number from an `mse` container.

| under `mse`, every NVFP4 sidecar line gains | meaning | source |
|---|---|---|
| `mse` | the written encoding | `src/main.rs:41-42` |
| `mse_ceil` | the same weights re-encoded with ceiling scales | `src/main.rs:42-43` |
| `mse_ratio` | the two above, as a ratio | `src/main.rs:43` |
| `max_abs_clipped` | a COUNT of clipped elements, those above six times the scale; the name is per report spec | `src/main.rs:43-44` |
| one `record: "section_summary"` line per NVFP4 section | aggregated numbers for `text`, `vit`, `ple`, `mtp` | `src/main.rs:45-46` |

- The GLOBAL tensor scale stays max-based in BOTH modes, so ladder utilization is unchanged; only the sub-block scale choice differs (`src/main.rs:47-48`).
- Cost of `mse`: about two to three times the per-value quantization work of `ceil` (`src/main.rs:49-50`).

## `requant-check` — the proof of the fetched originals (issue #76)

```
converter requant-check <dense.safetensors> <container.cnq> [--threads N] [--limit N]
```

- Added 2026-09-18 as an ADDITIVE read-only subcommand. It writes nothing, and a run without the word `requant-check` parses, reads and writes exactly what it did before: the conversion path is untouched (`src/main.rs:60-65`).
- It reads a safetensors file of ORIGINAL tensors — the one `../tools/fetch-dense-originals.py` fetches back by HTTP range — runs every tensor through the same `quantize_nvfp4` the conversion runs, and compares the produced NVFP4 blocks and the `f32` global scale against what the container stores under that name.
- The comparison is on bytes: per tensor it reports how many `36` B blocks differ, the first one that does, and whether its four scale bytes or its packed nibbles are what disagree. The global scale is compared on its bits.
- Scale mode: `mse` by default, because that is the mode `Qwen3.8-Flash-Next-CNQ4.5-M.cnq` was built with (`../docs/model-card.md`, Provenance). `--scales ceil` before the subcommand overrides it.
- It derives the tensor list a second time, from the container's own index (`section` `text`, `dtype` `nvfp4`, name without `.mlp.experts.`), and refuses if the fetched file is missing one of them. The fetch tool derives the same list from the sidecar — two independent derivations of one list.
- Exit code: `0` only if every tensor is identical including its global scale, `1` otherwise, `2` on a malformed argument or an unreadable file.
- What it proves and what it does not: see `../docs/dense-originals.md`.

## Tests

```
cd converter
cargo test --release
```

- Seven `#[test]` functions in `src/main.rs`, counted 2026-09-17, plus five in `src/requant_check.rs` added on 2026-09-18 (issue #76), so `cargo test --release` in this crate reads 12 passed, 0 failed on 2026-09-18.
- 2026-09-26 (Crow #300 C6): `cargo test` reads 57 passed, 0 failed (46 before C6; the eleven new ones cover the recipe rows, the index v2 round trip against `../engine/tests/fixtures/synthetic-v2/`, provenance and the per-family layer-rule arms).
- 2026-10-08 (crow-nest #154/#155): `cargo test --release` reads 77 passed, 0 failed (61 at `7c749d2`; the sixteen new ones cover the FP8 table against torch `float8_e4m3fn`, the dequant against DeepSeek's formula and transformers `Fp8Dequantize` (`tests/fixtures/fp8_fixtures.py`, oracle venv), the GLM row on all 76,108 tensors, the whitelist refusals, Huffman sizes, histogram sums, expert alignment, kill and resume, `--consume` and `--headers`).
- 2026-10-08 (crow-nest #156): `cargo test --release` reads 83 passed, 0 failed (77 before; the six new ones cover the `--layers` spec and its keep rule, a partial conversion of the miniature (named layers only, bytes equal to the full conversion, layer 44 filtered not missing, absent layers refused), `dequant_nvfp4` against gate 0's error sums bit for bit with a swapped-nibble and a swapped-scale control, and `dequant` on container records incl. row ranges).
- 2026-10-09 (crow-nest #182): `cargo test --release` reads 98 passed, 1 ignored (91 passed, 1 ignored before, at `82390fd` after #181; the seven new ones cover the MUL1 records of the miniature (size, 4096 alignment, `mul1::write_record` bytes, index entries under the engine's rule, MTP section), the dense part against `cnq4.5-glm5-next`, kill and resume at six points, `--mul1-wait`, `--layers 3`, the refusals, and the GLM store's 9,474,048-B record).
- They are not part of the engine's count: `cd engine && cargo test --release` reads 165 passed, 0 failed on 2026-09-17 and covers `crow_nest_engine` and `bin/serve` only. `tools/gate-linux.sh` pins THAT count, not this one, so a test added here moves no gate value.
