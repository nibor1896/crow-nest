# The oracle-KLD instrument at long context, and the English-prose corpus (issue #90)

`docs/oracle-kld.md` is the instrument of #78: KLD and same-top-1 of an arm against an
f32 oracle, in llama.cpp's own units, over two English code prompts of 298 and 607
tokens. This page is #90: the extension of that instrument to the regime where the
model actually fails - the 158k-163k prompt tokens of issue #68 - plus, after the
2026-09-20 scope change, an ENGLISH long-prose corpus (the German-prose component of
the original issue text was dropped by robin's direct order; nothing German was built).

What #90 delivers, in one list:

| piece | where |
|---|---|
| the clean 178,553-id history as VERIFIED token ids | `decode_out/oracle-longctx/longctx-170k-p{1..5}-ids.json` |
| the sampled teacher-forced row plan (6 depths x 64 rows) | `decode_out/oracle-longctx/row-plan.json` |
| the chunked, resumable f32 oracle for six-digit T | `oracle/ref_longctx_logits.py` |
| sparse dump support, per-depth/per-corpus summaries, the KLD-vs-position curve | `tools/oracle-kld.py` (extended) |
| the English long-prose corpus, tokenized | `tools/corpora/`, `decode_out/oracle-en/` |
| the crow-engine arm (paired-baseline mode) | `tools/oracle_longctx_engine_arm.sh` |
| the llama arm harness (one command when the GGUF is whole) | `tools/oracle_longctx_llama.sh` |

## 1. The form, and why its ids are trustworthy

The blind spot of #78 was REGIME: both references sit at a few hundred tokens, where
"the QSA indexer selects densely by construction". #68 left the raw material for the
long form: a CLEAN agentic history of 178,553 prompt tokens that answers 4 of 5 probes
without degeneration (`docs/long-context-goalmode.md` 4). That history is not stored
as ids anywhere - it is a pure function of `tools/longctx-gate.py`'s deterministic
generators (167 synthetic `libghost` files, 669 messages) and the model's chat
template.

`tools/oracle_longctx_rows.py ids` rebuilds it and VERIFIES the rebuild against the
only independent record: the per-probe `prompt_tokens` of the recorded gate run
`decode_out/68/longctx-170k.json`. All five probe prompts reproduce EXACTLY -

| probe | rebuilt ids | recorded prompt_tokens |
|---|---|---|
| 1 | 178,553 | 178,553 |
| 2 | 178,608 | 178,608 |
| 3 | 178,664 | 178,664 |
| 4 | 178,728 | 178,728 |
| 5 | 178,798 | 178,798 |

- which checks the material generators, the template (`enable_thinking=False`,
`add_generation_prompt=True` - 36 ids fewer than the thinking form) and the tokenizer
at once. A mismatch aborts and writes nothing; the first version of this tool caught
its own bug that way. The ids file used by the instrument is probe 1's: the recorded
178,553.

`tools/oracle_longctx_rows.py plan` samples CONTIGUOUS 64-row blocks at six depths -
contiguous because both arms read a block as one growing prefix:

| anchor | rows | prefix ids | why this anchor |
|---|---|---|---|
| 1000 | 937..1000 | 1,001 | the shallow control (dense QSA, like #78's sets) |
| 2564 | 2501..2564 | 2,565 | the FIRST SPARSE rows: T crosses the 2048 indexer budget |
| 50000 | 49937..50000 | 50,001 | the #68 healthy band |
| 100000 | 99937..100000 | 100,001 | mid window |
| 158000 | 157937..158000 | 158,001 | the degeneration band's lower edge (158,639 healthy) |
| 178553 | 178488..178551 | 178,552 | the live context size |

Row r is the distribution conditioned on ids[0..r] inclusive, predicting ids[r+1];
valid rows are 0..T-2. The plan file is also a `--row-groups` input to
`tools/oracle-kld.py`, which is what makes the per-depth summary below.

## 2. The f32 oracle at six-digit context - and the checkpoint gap

`oracle/ref_engine_logits.py` cannot run at 178k: it builds a [1,1,T,T] additive mask
(127 GB at this T). `oracle/ref_longctx_logits.py` is the same reference - same
transformers 5.16.1 eager f32 modules, same PLE row-gather math with the layer at
index 1 - fed in CHUNKS with caches:

- a `DynamicCache` for the 12 full-attention layers and their QSA indexer keys: the
  indexer still scores every query against the WHOLE prefix and keeps its top-512
  blocks, f32, exactly as the architecture defines it;
- the linear-attention layers' conv and recurrent states (transformers' own chunked
  gated-delta path);
- the PLE dilated-conv carry, with the n-gram uid table computed once per sequence by
  a vectorized port that is BIT-IDENTICAL to the original Python loops (the self-test
  diffs them, EOS segments included);
- logits computed only at the planned rows, written as a SPARSE dump (`out.f32` +
  `out.rows.json`).

**`self-test` proves the chunking** (tiny random checkpoint through the same loader,
257 tokens, chunk 64): the uid table matches the original loops exactly; the chunked
path's logits against the UNCHUNKED path's differ by at most **2.1e-07** with the
same top-1 on 256/256 rows; a run resumed from a mid-sequence checkpoint is
identical to the uninterrupted one; and the production path itself - `run` with a
plan, interrupted after two chunks, resumed - produces a dump that is
**byte-identical** to the uninterrupted `run`. One placement bug was found and fixed
by this test: the PLE injection sits between layers 0 and 1 (`ref_engine_logits.py`'s
placement), not before layer 0 - with it misplaced the delta was 7.7e-02 and 19
top-1 flips, which is what the self-test is FOR.

**The oracle cannot RUN here today, and that is measured, not assumed.**
`oracle/ref_longctx_logits.py check-weights` reports:

```
index total_size 359,999,963,128 B
shard files      131 referenced, 0 on disk, 131 MISSING
text tensors     1294 required by the oracle, 1294 unloadable
PLE shards       33 referenced, 0 on disk
```

The original Qwen/Qwen3.8-Flash-Next safetensors are gone machine-wide: the Linux
tree holds only `models/.../dense/dense.safetensors` (the 495 dense originals of
#76) plus expert MANIFESTS without data; the HF caches (Linux and the read-only
Windows mount) hold only GGUFs; the Windows tree's model directory is 23 MB. The
Linux disk has ~26 GB free against ~360 GB of shards, so a re-download is not a
night's option either. `run` refuses with this manifest rather than guess. robin's
options:

- **(A)** re-download the 360 GB original (479 GB free on the Windows NTFS partition;
  robin's call) or free Linux space;
- **(B)** produce the new f32 rows on another box that still has the checkpoint -
  `oracle/ref_longctx_logits.py run --ids ... --plan ... --out ...` is the whole
  command, resumable at chunk granularity (`--save-every`, a `.state.pt` beside the
  dump);
- **(C)** proceed WITHOUT new f32 rows tonight - paired-baseline mode (section 3) -
  which is what is running.

For the day the checkpoint returns, `verify` replays the EXISTING references row by
row with the chunked path and reports the deltas, so a new machine's rows can be
trusted against the 2026-09-05 Windows dumps:

```
.venv-oracle/bin/python oracle/ref_longctx_logits.py verify \
    --ids decode_out/oracle-tf298/tf298-ids.json \
    --against decode_out/oracle-tf298/ref-logits.f32 --rows 40
```

Runtime estimate for the 178k sweep on this machine (24-thread CPU): the 12
full-attention layers dominate at ~2e15 f32 FLOPs of QK/PV plus the indexer's Python
per-query loop; at 0.3-1 TFLOPS sustained that is roughly 4-12 hours for ONE pass
over all 178,553 tokens - which is exactly why the runner is chunked, resumable, and
writes only the planned 384 rows (6 x 64) of logits.

## 3. Tonight's reference: paired-baseline mode

The estimator needs a reference; without the f32 oracle the DESIGNATED BASELINE is
the engine's default of record - the `none` arm - and every question becomes PAIRED,
on the same rows, in the same units:

```
tools/oracle-kld.py --ref <none sparse dump> --arm kvbf16=<sparse dump> \
    --arm mma0=<...> --paired kvbf16,none --row-groups row-plan.json \
    --kld-vs-position 6 --json out.json
```

This is the same methodology as `docs/oracle-kld.md` 6: `kvbf16` (bf16 KV, strictly
more precision than FP8 E4M3) and `mma0` (f32 activations) sit at `none` or a shade
closer to f32; their spread at 100k+ IS the long-context noise floor, and the
existing short-context floor (0.025 mean KLD / 1.5 pp top-1 paired on 607 rows) is
the provisional gate until the long-context floor is measured. What this mode can
decide: whether a quality-touching change (#T6/#T7/#T9) moves the engine's
distributions at depth, relative to its own default, beyond the floor. What it
cannot: absolute distance to f32 - that waits for the checkpoint (options A/B).

**The crow arm has the same collect-all ceiling as the oracle had.** `decode parity`
writes EVERY row of its ids file, so anchor 50000 would be a 50 GB dump;
`tools/oracle_longctx_engine_arm.sh` therefore runs the two feasible anchors
(1000 and 2564 - the dense control and the FIRST SPARSE rows), refuses the deep
anchors with the reason, and trims every dump to the plan's 64 rows (the dense file
is hashed then deleted; `SHA256SUMS` records it). The deep-anchor engine arm needs a
`--rows` flag in `decode parity` - an engine/ change outside this issue's file
ownership. The script takes the GPU flock (`/tmp/crow-gpu.lock`), waits out any live
engine, and retries a ladder of pinned budgets with a 10-minute backoff for up to 8
hours, because the fleet shares this machine's RAM (2026-09-20: 10-14 GiB free
against a loader that wants 16-44 GiB pinned).

**The llama arm is a harness, not a run**: shard 1 of the Unsloth UD-Q2_K_XL GGUF is
broken tonight. `tools/oracle_longctx_llama.sh` is the one command for when it is
whole - round-trip check, then `llama-row-probs --no-cache-prompt` over the block,
then the sparse trim. It carries the same >2564-row ceiling
(`llama-row-probs.py` writes rows 0..last contiguously).

## 4. The English long-prose corpus

The non-word symptom was reference-free until now; this gives it a logits-space form
in English. `tools/oracle_longctx_corpus.py fetch` pulls five public-domain books
from Project Gutenberg into `tools/corpora/` (raw + stripped + `SOURCES.json` with
sha256s and the exact strip rule); `emit` tokenizes them with the ORIGINAL tokenizer
- raw text, `add_special_tokens=False`, NO chat template, the corpus form - and
writes one contiguous 513-id slice per book, centered on the book's midpoint:

| text | book ids | slice | rows |
|---|---|---|---|
| moby-dick | 311,730 | 155,609..156,121 | 512 |
| pride-prejudice | 174,832 | 87,160..87,672 | 512 |
| frankenstein | 99,078 | 49,283..49,795 | 512 |
| dracula | 216,348 | 107,918..108,430 | 512 |
| huckleberry-finn | 156,424 | 77,956..78,468 | 512 |

2,560 teacher-forced prose rows, five per-corpus groups in
`decode_out/oracle-en/en-row-groups.json`. When the checkpoint returns, the f32
reference for them is one command per text (`run --ids ... --all-rows`), and the
reading is `oracle-kld.py` with `--row-groups en-row-groups.json` - the per-corpus
summary.

## 5. The reading, end to end

```
# the form (ids verified, plan written) - CPU, seconds
.venv-oracle/bin/python tools/oracle_longctx_rows.py ids
.venv-oracle/bin/python tools/oracle_longctx_rows.py plan --depths 1000 2564 50000 100000 158000 178553

# the f32 oracle (ONLY where the checkpoint exists - check-weights says which)
.venv-oracle/bin/python oracle/ref_longctx_logits.py check-weights
.venv-oracle/bin/python oracle/ref_longctx_logits.py run \
    --ids decode_out/oracle-longctx/longctx-170k-p1-ids.json \
    --plan decode_out/oracle-longctx/row-plan.json \
    --out decode_out/oracle-longctx/ref/gpu-logits.f32 --chunk 2048

# the crow engine arm (GPU flock; feasible anchors only tonight)
nohup tools/oracle_longctx_engine_arm.sh > decode_out/oracle-longctx/engine-arm.log 2>&1 &

# the llama arm (when the GGUF is whole)
tools/oracle_longctx_llama.sh

# the reading - sparse dumps are auto-detected by their .rows.json sidecar
tools/oracle-kld.py --ref decode_out/oracle-longctx/ref/gpu-logits.f32 \
    --arm none=decode_out/oracle-longctx/engine/a2564/none/plan-rows.f32 \
    --row-groups decode_out/oracle-longctx/row-plan.json \
    --kld-vs-position 6 --paired none,ref \
    --json decode_out/oracle-longctx/kld-90.json
```

The sparse reader is validated against the REAL existing dumps: rows of
`decode_out/oracle-tf298` carved to a sparse subset read back with per-row KLD
diff **exactly 0.0** against the dense read of the same rows
(`decode_out/oracle-longctx/validate/`). `python3 tools/test_oracle_kld.py`
(55 tests, unchanged) and `python3 tools/oracle_longctx_test_kld.py` (9 tests,
new: sparse equivalence, row groups, the curve, the subset tool) are both green.

## 6. What this does and does not say

- The 178,553-id history is now a REPRODUCIBLE artifact, not a recorded number -
  five exact id-count matches against the only independent record.
- The instrument covers the failure REGIME in form: rows at 158k and 178k exist as
  a plan, the runner exists and is proven equivalent at short T, and every arm has
  a one-command path. What does NOT exist yet, honestly: f32 reference rows at
  depth (checkpoint gone), engine/llama dumps at depth (collect-all designs), and
  therefore any KLD number at 100k+. No number in this document is invented to
  fill that gap.
- The first sparse-QSA rows (anchor 2564) are within reach of TONIGHT's engine arm
  - the exact boundary where "dense by construction" ends has never been measured
  against anything.
- English prose is now a corpus of the instrument; the German symptom stays what
  `docs/quality-probe.md` measures it with (reference-free counts), per the scope
  change - no German corpus was built.
