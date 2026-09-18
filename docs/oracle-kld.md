# The f32 oracle, and what a quantization costs against it

`tools/oracle-kld.py` is step 4 of the requant series (issue #78), after the characterization
harness of #75, the BF16 originals of #76 and the dense overlay of #77. It changes nothing in
the engine, the converter or the container. It turns a measurement that existed as a
throwaway script into a tool of the repository, and it puts the second quant — the Unsloth
GGUF this project keeps comparing itself against — on the SAME reference rows.

- `docs/quality-probe.md` — the reference-FREE reading of answer quality, and why it cannot
  say how far a quant is from the original.
- `docs/dense-overlay.md` — the arms `control` and `orig` below, and what they are.
- `docs/dense-originals.md` — where the BF16 originals came from.

## 1. What is measured, and in whose units

The literature's reading of a quantization is reference-based: teacher-force the quantized
model and the unquantized one over the same token sequence and compare the two distributions
per position (arXiv 2407.09141, llama.cpp discussion 4110). llama.cpp prints exactly that
under `llama-perplexity --kl-divergence`, and its README scoreboard is the scale everybody
quotes. This tool reports the same statistics, computed the same way, so that a number here
can be laid beside a row of that table.

| what | definition | where it comes from |
|---|---|---|
| KLD | `sum_i p_ref(i) * (log p_ref(i) - log q_arm(i))` over every `i` with `log p_ref(i) > -16`, natural log | `tools/perplexity/perplexity.cpp:222-231` |
| same top-1 | the share of rows on which the arm's argmax is the reference's argmax, with `sqrt(p(1-p)/(n-1))` | `perplexity.cpp:2004-2005` |
| uncertainty on a mean | `sqrt((sum2/n - mean^2)/(n-1))`, and ZERO for ten rows or fewer | `perplexity.cpp:1770-1778` |
| percentile | linear interpolation at `fraction*(n-1)`, not a nearest rank | `perplexity.cpp:1954-1960` |
| median | the mean of the two middle values when `n` is even | `perplexity.cpp:1951` |

Read the direction: **the reference is the first argument.** `KL(P_reference || Q_arm)` asks
what it costs to describe the f32 model's next-token distribution with the quantized model's,
and it is not symmetric. The source is the one on this machine,
`~/.local/share/crow/src/llama.cpp`, pin `6c84c7d5d` plus PR #27880 and PR #28040 — the tree
`~/.local/share/crow/bin/llama-server` was built from.

ONE THING IS NOT llama.cpp's. Its `Δp` is the change in the probability of the CORPUS's next
token. There is no corpus here — only a token sequence the reference was teacher-forced on —
so this tool's `dp` is the change in the probability the arm gives the REFERENCE'S top-1
token, and the two columns that matter are the share of rows that lost more than 10 and more
than 50 percentage points of it. The report says so on every run.

## 2. The two references, and what they are NOT

Both were produced on **2026-09-05** by `oracle/ref_engine_logits.py` — transformers 5.16.1,
eager, **f32**, the full 48 layers, **with the PLE layer at index 1** — on the CPU, and they
are the same files the vault note
`die-degeneration-war-der-ple-schritt-im-decode-pfad-nicht-der-container` was written from.
They were copied to this machine from the Windows tree
(`/run/media/.../dev/crow-nest/decode_out/oracle-t2w/`, mounted read only) on 2026-09-19 and
byte-verified by sha256 against the source.

| | `decode_out/oracle-tf298/` | `decode_out/oracle-t2b/` |
|---|---|---|
| `ref-logits.f32` | 299,970,560 B = 302 rows x 248,320 f32 | 606,894,080 B = 611 rows x 248,320 f32 |
| sha256 | `3268a4515b95519dc0315ebe91c8e229e6330c048822cfe3daee24acdf9c8ed0` | `6de6967cf805f4655e655aeb41f170fe9bac311737e63277e835d47d8011f682` |
| ids | `tf298-ids.json`, 298 | `t2b-tf-ids.json`, 607 |
| text | the t2-write prompt (235 tokens, "write a fixed-capacity LRU cache header in C++17") plus the 63 tokens the engine generated on 2026-09-05 | the t2b-write-refactor prompt (596 tokens, `t2b-prompt.txt`, "refactor this C++ function") plus the 11 tokens the engine generated before its EOS |
| rows that are teacher-forced ON THOSE IDS | 0..297 | 0..606 |
| the other rows | 298..301 | 607..610 |

**Why the last four rows of each file are excluded.** `ref_engine_logits.py` teacher-forces
over `all_ids[:rows]`, and `all_ids` beyond the id file is what the ENGINE greedily produced
on 2026-09-05 — `[248045, 846, 248046, 198]` and `[198, 248044, 248046, 198]`. Today's engine
continues differently (`decode parity` writes `[74455, 198, 248068, 198]` for the first set),
so those four rows of an arm are conditioned on a different context than the same four rows
of the reference. They are not comparable and every run below stops before them. That is why
this document says 298 and 607 rows where the files hold 302 and 611.

**What these references are NOT.** They are not a corpus. They are two English code tasks, one
short and one medium, in the chat template's raw ids with no template applied on top. The
context is 298 and 607 tokens, which is SHORT: the QSA indexer selects densely by
construction at that length — the 2026-09-05 run records it in its own header,
`"production FP4 weights, FP8-KV, QSA dense (short T)"` — so nothing here reads the sparse
attention regime the operating point actually runs in. They are not perplexity either — no
number below is a perplexity, and `llama-perplexity`'s PPL columns have no counterpart here,
because a 298-token sample of one task is not a perplexity measurement.

## 3. The tool

```
tools/oracle-kld.py --ref decode_out/oracle-tf298/ref-logits.f32 --rows 0:298 \
    --prompt-rows 235 \
    --arm none=decode_out/oracle-tf298/none/gpu-logits.f32 \
    --arm orig=decode_out/oracle-tf298/orig/gpu-logits.f32 \
    --paired orig,none --per-row out.json --json summary.json
```

stdlib only — there is no numpy on this machine — and no GPU, no server and no model. It
reads flat `rows x vocab` f32 little-endian dumps, which is what `decode parity` writes and
what the 2026-09-05 oracle was stored as, one row at a time. 298 rows and seven arms cost 34 s;
607 rows and seven arms cost 69 s. `tools/test_oracle_kld.py` is 55 unit tests of the pure
functions — the divergence by hand on a two-symbol and a four-symbol distribution, the
direction, the support cut, llama.cpp's percentile and uncertainty formulas, the sign test,
the row range and an end-to-end run over a four-row eight-token reference written to real
files.

**The truncated estimator, which in the end was not needed.** A second arm kind exists,
`--topn-arm`, for a model that can only report its top N probabilities; `--truncate M` then
restricts the sum to the reference's M most probable tokens and `--arm-topn N` gives a full
dump the same blindness, so the two kinds run the SAME estimator. A token the arm cannot
answer for is charged at the arm's smallest known probability, which makes the result a lower
bound and never an estimate from above. Section 4 is why the llama.cpp arm did not need it.

## 4. The llama.cpp arm: the route, and the route that does not exist

The arm is the Unsloth `Qwen3.8-Flash-Next-UD-Q2_K_XL` GGUF — the quant robin's good poster run
uses — served by `~/.local/share/crow/bin/llama-server` at the operating point of record
(`~/.local/share/crow/tools/start-server.py flash-next-q2-k-xl`, port 8083, build
`b52-cbca449`, `-c 200000 -b 2048 -ub 2048 -ctk q8_0 -ctv q8_0 -ncmoe 31 -t 24 --fit off
--load-mode mmap -np 1 --jinja`). One engine on the card at a time; nothing else ran while it
did.

### 4.1 `llama-perplexity` cannot read these rows, and that is measured, not assumed

The obvious route is to build `llama-perplexity` from the same tree and let it dump logits
with `--kl-divergence-base`. It is not available, for a reason that has nothing to do with the
build: **that tool reads a TEXT file.** It tokenizes it with `common_tokenize(ctx, prompt,
true)` (`tools/perplexity/perplexity.cpp:310`), and the fourth argument `parse_special`
defaults to FALSE (`common/common.h:1050`). So the route exists only if the 298 ids survive
detokenize -> tokenize. They do not. `tools/llama-row-probs.py --round-trip` asked the running
server, through `/detokenize` and `/tokenize`, in all four settings:

| `/tokenize` setting | result |
|---|---|
| `parse_special=false` (llama-perplexity's), `add_special` either way | 368 tokens instead of 298; the first id already differs — `<\|im_start\|>` (248045) comes back as the plain text tokens `27, 91, 316, ...` |
| `parse_special=true`, `add_special` either way | 294 tokens instead of 298; first difference at position 239, where the reference's three separate newlines `198, 198, 198` come back as the single merged token `14200` |

Two independent failures, and the second one is the interesting one: even with special tokens
parsed, BPE re-merges a run of newlines that the reference carries unmerged, so no text file
exists whose tokenization is this id sequence. The route is INVALID, not inconvenient, and
`llama-perplexity` was therefore never built. The check is one command against a running
server and costs no GPU time worth naming.

### 4.2 What `llama-server` will do instead

`/completion` takes a prompt as an ARRAY OF TOKEN IDS and uses it verbatim —
`tokenize_input_subprompt`, `tools/server/server-common.cpp:972-975`: no BOS is added (the
GGUF says `tokenizer.ggml.add_bos_token = false` anyway) and no chat template is applied. So
the distribution at row `r` is one request whose prompt is `ids[0..r]` and whose `n_predict`
is 1, and `tools/llama-row-probs.py` is 298 or 607 such requests.

**The probabilities are the model's and not the sampler's.** With `post_sampling_probs` false
— the default, and sent explicitly — the server runs `get_token_probabilities`
(`tools/server/server-common.cpp:1501`): a plain softmax over the WHOLE vocabulary of the
pre-sampling logits, then a partial sort. Temperature, top-k, top-p and min-p never touch
them. `populate_token_probs` (`server-context.cpp:1932`) takes the top `n_probs` of that.

**The whole vocabulary, not a top N.** `n_probs` carries no cap in the request schema
(`tools/server/server-schema.cpp:178`), and asking for all of it is cheap on this machine:
measured 2026-09-19 on a warm slot, `n_probs` 1 / 2,000 / 20,000 / 60,000 / 248,320 cost
0.06 / 0.07 / 0.10 / 0.17 / **0.47 s** and returned 3 KB / 178 KB / 1.9 MB / 5.9 MB / 24.8 MB.
So there was no reason to accept a truncated estimator: the arm is collected in FULL and enters
`tools/oracle-kld.py` as an ordinary `--arm`, through the same code path and the same
untruncated sum as every crow-nest arm. `tools/llama-row-probs.py` writes the natural
logarithms of the probabilities into the same flat `rows x vocab` f32 file `decode parity`
writes; a log-probability row and a logit row are the same thing to a tool that subtracts the
row's own log-sum-exp.

**Three checks that the arm is the arm.**

- **The vocabulary is the same vocabulary.** Asked for 300,000 entries, the server returned
  exactly **248,320**, ids 0 to 248,319, all distinct — the reference's column count to the
  token. The GGUF header agrees: `tokenizer.ggml.tokens` is an array of 248,320.
- **PLE is in the graph.** The GGUF carries the architecture's PLE keys —
  `qwen4exp.ple.layers = [1]`, `ple.ngram_size 3`, `ple.heads_per_ngram 8`,
  `ple.conv_kernel 4`, the 16 head offsets and vocab sizes — and llama.cpp builds it:
  `src/models/qwen4exp.cpp:332-334` calls `build_ple` for every layer `hparams.is_ple(il)`
  names, with `ple_key`, `ple_value`, `ple_norm_{key,query,conv}` and `ple_conv1d` created at
  `qwen4exp.cpp:193-198`. Layer 1, the same layer crow-nest puts it at.
- **llama.cpp's own softmax is not exactly normalized, and it does not matter.** Its f32
  accumulation over a quarter of a million terms returns a mass of 0.99994 to 1.0027 rather
  than 1. `tools/oracle-kld.py` subtracts each row's log-sum-exp, which renormalizes it away
  exactly.

### 4.3 The prompt cache is not free, so the arm does not use it

With `cache_prompt` true each row is one decode step on top of the previous row's KV; without
it the whole prefix is prefilled again and the row is a pure function of its prefix. **The two
do not agree.** Measured 2026-09-19 over the full vocabulary:

| row | `KL(uncached \|\| cached)` | same top-1 |
|---|---|---|
| 5 | 0.012275 | NO |
| 40 | 0.027742 | yes |
| 120 | 0.000910 | yes |
| 235 | 0.000423 | yes |
| 297 | 0.000001 | yes |

That is a batch-size effect in ggml's own arithmetic — the same class of thing
`docs/architecture.md` 8.7 records for crow-nest's NVRTC drift — and at rows 5 and 40 it is as
large as the differences between two quantization arms. So the arm of record is collected
with `--no-cache-prompt`: every row prefilled from scratch, 184,756 tokens for the 607-row
set. It costs 7.6 minutes for 298 rows and 18.7 minutes for 607 against 3.6 and 7.3 with the
cache, and it buys a row that depends on nothing but its own prefix. Section 6 keeps the
cached collection as a second arm, because the distance between them is this arm's noise floor
and belongs in the table rather than in a footnote.

## 5. The arms, and the two tables

Seven arms, all measured 2026-09-19 on RTX 5090 / Arch Linux, one engine on the card at a
time, the crow arms at repo commit `09cb00a`.

| arm | what it is |
|---|---|
| `none` | `CNQ4.5-M` at the default of record — NVFP4 everywhere, FP8 E4M3 KV cache, the NVFP4 activation cascade, `CROW_GRAPH=1 CROW_MMA=1` |
| `control` | `none` plus the DEQUANTIZED-control overlay (#77): the container's own NVFP4 dense values widened to bf16. Carries no new information |
| `orig` | `none` plus the ORIGINALS overlay (#77): the 495 dense text tensors as the BF16 originals of #76 |
| `kvbf16` | `none` with `CROW_KV=bf16` — the KV cache in bf16 instead of FP8 E4M3, i.e. strictly MORE precision |
| `mma0` | `none` with `CROW_MMA=0` — f32 activations instead of the NVFP4 cascade, again strictly more precision |
| `llama` | Unsloth `Qwen3.8-Flash-Next-UD-Q2_K_XL` through `llama-server`, every row prefilled from scratch (section 4) |
| `llama-cached` | the same GGUF, every row one decode step on the previous row's KV — the control of section 4.3, not a second quant |

### 5.1 The 298-row set (`tf298`, the t2-write prompt)

| arm | same top-1 as f32 | mean KLD | median | p90 | p99 | p99.9 | max |
|---|---|---|---|---|---|---|---|
| `none` | 82.89 ± 2.19 % | 0.3431 ± 0.0508 | 0.0717 | 0.818 | 3.528 | 9.211 | 9.527 |
| `control` | 80.54 ± 2.30 % | 0.3491 ± 0.0520 | 0.0656 | 0.972 | 2.899 | 9.588 | 9.995 |
| `orig` | **84.23 ± 2.11 %** | **0.3233 ± 0.0431** | 0.0662 | 0.798 | 2.915 | 6.772 | 6.945 |
| `kvbf16` | 81.54 ± 2.25 % | 0.3627 ± 0.0505 | 0.0769 | 0.895 | 3.024 | 8.919 | 9.252 |
| `mma0` | 81.54 ± 2.25 % | 0.3323 ± 0.0521 | 0.0660 | 0.841 | 3.230 | 9.698 | 10.169 |
| `llama` | 82.21 ± 2.22 % | 0.4083 ± 0.0585 | 0.0744 | 1.021 | 5.440 | 8.955 | 9.353 |
| `llama-cached` | 82.55 ± 2.20 % | 0.4290 ± 0.0612 | 0.0806 | 1.228 | 5.650 | 9.581 | 10.012 |

Prompt rows 0..234 against answer rows 235..297:

| arm | same top-1, prompt / answer | mean KLD, prompt / answer |
|---|---|---|
| `none` | 82.55 ± 2.48 % / 84.13 ± 4.64 % | 0.3668 ± 0.0632 / 0.2549 ± 0.0460 |
| `control` | 78.30 ± 2.69 % / 88.89 ± 3.99 % | 0.3842 ± 0.0650 / 0.2183 ± 0.0398 |
| `orig` | 82.13 ± 2.50 % / 92.06 ± 3.43 % | 0.3667 ± 0.0528 / 0.1614 ± 0.0490 |
| `kvbf16` | 80.43 ± 2.59 % / 85.71 ± 4.44 % | 0.3940 ± 0.0623 / 0.2461 ± 0.0536 |
| `mma0` | 80.00 ± 2.61 % / 87.30 ± 4.23 % | 0.3616 ± 0.0651 / 0.2231 ± 0.0414 |
| `llama` | 80.85 ± 2.57 % / 87.30 ± 4.23 % | 0.4297 ± 0.0712 / 0.3284 ± 0.0768 |
| `llama-cached` | 80.85 ± 2.57 % / 88.89 ± 3.99 % | 0.4616 ± 0.0751 / 0.3073 ± 0.0732 |

Every arm is closer to the oracle on the answer rows than on the prompt rows. That is not a
property of the quant: the answer is 63 tokens the ENGINE itself produced greedily, so it is
text this model finds easy, and the reference agrees with it more sharply there.

### 5.2 The 607-row set (`t2b`, the write-refactor prompt)

| arm | same top-1 as f32 | mean KLD | median | p90 | p99 | p99.9 | max |
|---|---|---|---|---|---|---|---|
| `none` | 82.37 ± 1.55 % | 0.4607 ± 0.0484 | 0.0507 | 1.189 | 5.767 | 10.643 | 11.560 |
| `control` | 81.38 ± 1.58 % | 0.4744 ± 0.0477 | 0.0586 | 1.217 | 6.357 | 9.862 | 10.480 |
| `orig` | **82.70 ± 1.54 %** | **0.3480 ± 0.0351** | 0.0371 | 1.009 | 4.288 | 7.346 | 9.443 |
| `kvbf16` | 82.04 ± 1.56 % | 0.4723 ± 0.0517 | 0.0502 | 1.199 | 6.947 | 11.727 | 11.949 |
| `mma0` | 81.71 ± 1.57 % | 0.4834 ± 0.0488 | 0.0635 | 1.362 | 5.813 | 10.456 | 12.714 |
| `llama` | 79.57 ± 1.64 % | 0.5571 ± 0.0576 | 0.0931 | 1.391 | 6.685 | 12.913 | 14.753 |
| `llama-cached` | 80.56 ± 1.61 % | 0.5085 ± 0.0520 | 0.0828 | 1.263 | 6.036 | 12.542 | 14.488 |

The answer half of this set is ELEVEN rows — the 11 tokens the engine produced before its
early EOS, one row above the threshold at which llama.cpp starts printing an uncertainty at
all (`count > 10`, `perplexity.cpp:1776`). The uncertainties say what eleven rows are worth:
`none` reads 81.82 ± 12.20 % and 0.3372 ± 0.1078, `llama` 90.91 ± 9.09 % and 0.2624 ± 0.0810.
It is in `decode_out/oracle-t2b/kld-78.json`, it is printed for completeness, and it decides
nothing. The prompt half is 596 of the 607 rows and carries every figure quoted from this set.

### 5.3 The reference's top-1 probability

`dp` is `q_arm(ref top-1) - p_ref(ref top-1)`; negative means the arm is less sure of the
token the f32 model wanted.

| arm | mean dp, 298 rows | lost > 10 pp | lost > 50 pp | mean dp, 607 rows | lost > 10 pp | lost > 50 pp |
|---|---|---|---|---|---|---|
| `none` | -7.53 ± 1.20 % | 29.9 % | 5.4 % | -6.91 ± 0.90 % | 25.7 % | 5.4 % |
| `control` | -7.77 ± 1.20 % | 27.9 % | 4.7 % | -7.18 ± 0.92 % | 28.2 % | 5.4 % |
| `orig` | -4.24 ± 1.29 % | 23.5 % | 5.4 % | -7.66 ± 0.85 % | 26.7 % | 5.9 % |
| `kvbf16` | -7.66 ± 1.25 % | 29.5 % | 5.4 % | -6.60 ± 0.91 % | 25.7 % | 5.4 % |
| `mma0` | -7.85 ± 1.15 % | 30.9 % | 5.0 % | -7.50 ± 0.94 % | 28.7 % | 5.6 % |
| `llama` | -5.72 ± 1.37 % | 26.5 % | 4.7 % | **-10.07 ± 1.01 %** | **32.5 %** | **7.1 %** |
| `llama-cached` | -5.96 ± 1.37 % | 28.5 % | 5.0 % | -9.35 ± 0.97 % | 31.5 % | 6.4 % |

### 5.4 Paired, per row

Two means 0.02 apart on 298 noisy rows are unreadable side by side. The same rows subtracted
from each other are not: `--paired A,B` gives the mean of `KLD(A) - KLD(B)` over the SAME row
with its standard error, and a two-sided exact sign test on how often A is the worse of the
two.

| A - B | 298 rows: mean difference | A worse on | sign test p | 607 rows: mean difference | A worse on | sign test p |
|---|---|---|---|---|---|---|
| `orig` - `none` | -0.0198 ± 0.0271 | 142/298 | 0.45 | **-0.1127 ± 0.0354** | 187/607 | **1.5e-21** |
| `control` - `none` | 0.0060 ± 0.0108 | 152/298 | 0.77 | 0.0137 ± 0.0142 | 350/607 | 1.8e-4 |
| `kvbf16` - `none` | 0.0196 ± 0.0092 | 172/298 | 0.0090 | 0.0116 ± 0.0148 | 287/607 | 0.19 |
| `mma0` - `none` | -0.0108 ± 0.0094 | 134/298 | 0.093 | 0.0227 ± 0.0116 | 381/607 | 3.3e-10 |
| `llama` - `none` | 0.0652 ± 0.0410 | 147/298 | 0.86 | **0.0964 ± 0.0545** | 377/607 | **2.6e-9** |
| `llama` - `orig` | 0.0850 ± 0.0424 | 175/298 | 0.0031 | **0.2091 ± 0.0526** | 430/607 | **2.9e-25** |
| `llama-cached` - `llama` | 0.0207 ± 0.0115 | 167/297 | 0.037 | -0.0486 ± 0.0254 | 249/606 | 1.3e-5 |

**The 298-row set cannot see the dense path and the 607-row set can.** `orig - none` is
-0.0198 ± 0.0271 on 298 rows, which is nothing, and -0.1127 ± 0.0354 with a sign test of
1.5e-21 on 607. #77 measured the same improvement against the unquantized goldens (8.8x closer
at layer 0) and could only show a 1.79-per-1000 move on the reference-free probe; here it is
a 24 % cut in mean KL divergence against the f32 model, and it is the largest real effect in
either table.

### 5.5 What the truncated estimator would have cost

Had `n_probs` been capped and the llama arm come back as a top 2,000 only, `--truncate 32
--arm-topn 2000` is the estimator both arms would have run. On the 298-row set it reads
`none` 0.3224 against the full 0.3431 and `llama` 0.3734 against 0.4083 — 6 and 9 % low, as a
lower bound must be. The reference's top 32 carry **97.75 %** of its mass on average, the arms
could answer for every one of them on 92 and 93 % of rows, and the paired difference survives at
0.0510 ± 0.0402 against the full 0.0652 ± 0.0410. It would have been usable. It was not needed.

## 6. The noise floor

**Run to run, at a fixed configuration, the noise is exactly ZERO.** `decode parity` was run
twice over each id set with identical environment and the dumps are byte-identical:
`eee41832c6cc…` twice for the 298-row set and `79a4d2b682b4…` twice for the 607-row set. No
difference in any table above can be explained by re-running.

**Configuration to configuration, it is not.** `control`, `kvbf16` and `mma0` are the arms
that settle the scale, because none of them can be WORSE than `none` for an information
reason: the control's bf16 is about 2^-9 away from the NVFP4 it dequantizes
(`docs/dense-overlay.md` 2), a bf16 KV cache is strictly more precise than FP8 E4M3, and f32
activations are strictly more precise than the NVFP4 cascade. Every one of them should
therefore be at `none` or a shade closer to f32. What is measured instead:

| set | mean KLD across `none`, `control`, `kvbf16`, `mma0` | same top-1 across the same four |
|---|---|---|
| 298 rows | 0.3323 to 0.3627, a band of **0.030** | 80.54 to 82.89 %, a band of **2.35 pp** |
| 607 rows | 0.4607 to 0.4834, a band of **0.023** | 81.38 to 82.37 %, a band of **0.99 pp** |

Paired per row against `none`, the same three arms reach **0.023** in mean difference (`mma0`
on 607 rows), and two of the three sign tests are "significant" at rows where the mean
difference is inside its own standard error — which is the honest reading of a sign test on
607 correlated rows and the reason both are printed.

**So**: on 298 rows a difference in mean KLD below about **0.03**, or in same top-1 below about
**2.5 percentage points**, says nothing. On 607 rows the thresholds are about **0.025** and
**1.5 points**. Paired, a mean per-row difference has to clear about **0.025** before it is more
than the instrument. `orig - none` at -0.1127 on 607 rows clears it by a factor of five;
`llama - none` at +0.0964 clears it by four.

**The llama arm carries one more source that the crow arms do not**, and it is in the table
rather than in a footnote: how a row is collected. `llama-cached - llama` is +0.0207 ± 0.0115
on 298 rows and -0.0486 ± 0.0254 on 607, and the same-top-1 share moves 0.34 and 0.99 points
between the two collections. That is the same order as the whole `llama - none` difference, so
the llama arm's distance to the crow arms is stated with both collections in section 7 and
neither is quoted alone.

## 7. What this says, and what it does not

### 7.1 On llama.cpp's own scale

For scale, the rows of `~/.local/share/crow/src/llama.cpp/tools/perplexity/README.md`
("LLaMA 2 vs LLaMA 3 Quantization comparison", revision `f364eb6f`, Wikitext-2, CUDA):

| quant | mean KLD | same top-1 |
|---|---|---|
| LLaMA 2 7B `q4_K_M` | 0.012686 ± 0.000079 | 94.665 ± 0.055 % |
| LLaMA 3 8B `q4_K_M` | 0.031273 ± 0.000238 | 91.901 ± 0.072 % |
| LLaMA 2 7B `q2_K` | 0.108903 ± 0.000645 | 85.584 ± 0.086 % |
| LLaMA 3 8B `q2_K` | 0.445132 ± 0.001835 | 71.138 ± 0.119 % |

Both quants measured here land in the `q2_K` band and nowhere near the `q4_K_M` one:
CNQ4.5-M at 0.343 / 0.461 and 82.4 to 82.9 %, UD-Q2_K_XL at 0.408 / 0.557 and 79.6 to 82.2 %.
That comparison is a SCALE and not an equality — a different model, a different corpus, a
different tokenizer and 100k-plus rows against 298 and 607 — and it is quoted here for the
order of magnitude only.

### 7.2 The finding that reframes the series

**UD-Q2_K_XL is not closer to the f32 model than CNQ4.5-M.** On the 607-row set it is
further: mean KLD 0.557 against 0.461 (0.509 against 0.461 if the cached collection is used
instead) and same top-1 79.57 ± 1.64 % against 82.37 ± 1.55 %. Paired on the same rows,
`llama - none` is +0.0964 ± 0.0545 — 1.8 standard errors, which alone would decide nothing —
but `llama` is the worse of the two on **377 of 607 rows**, and a 62 : 38 split of 607 rows is
a sign test of 2.6e-9. It is the sign test that carries this, not the mean. On the 298-row
set the two cannot be told apart at all (+0.0652 ± 0.0410, sign test 0.86, 147 of 298).
Against the BEST crow arm, the BF16 dense overlay of #77, the gap is larger and is not close:
+0.2091 ± 0.0526, `llama` worse on 430 of 607 rows.

That matters because of what #75 measured on the same two engines: llama-server writes 8.21
German non-words per 1000 on long prose where `serve` writes 16.40, and its broken words are
of a different kind entirely (`docs/quality-probe.md` 5 and 7). **Those two readings point
opposite ways, and both are measurements.** So the answer-quality gap between the two engines is NOT a
distance-to-f32 gap on English code text, and the story "CNQ4.5-M is a worse quant than the
Unsloth GGUF" does not survive contact with the literature's own metric. What differs between
the two engines also includes the tokenizer path, the chat template, the sampler and the
language — this instrument holds none of that fixed, and section 7.4 says what it can decide.

### 7.3 What is ruled out as the main source of CNQ4.5-M's distance

Each of these arms removes one suspect and leaves the distance where it was:

- **the dense weights are not the main source.** The BF16 originals over all 495 dense text
  tensors cut the mean KLD from 0.4607 to 0.3480 on 607 rows — real, four times the noise
  floor, and a 24 % cut. Three quarters of the distance stays.
- **the FP8 KV cache is not a source at all.** `kvbf16` doubles the KV precision and moves the
  mean KLD by +0.0116 ± 0.0148 on 607 rows, inside the floor and in the wrong direction.
- **the NVFP4 activation cascade is not a source.** `mma0` computes the dense projections on
  f32 activations and moves +0.0227 ± 0.0116, again inside the floor and in the wrong
  direction.
- **the engine is not re-deciding anything between runs.** Two runs of the same arm are the
  same bytes.

### 7.4 What is NOT separated

- **The routed experts and the NVFP4 PLE table are still one suspect.** Nothing here touches
  either. 97 % of the container's bytes are routed experts at RTN NVFP4, and the PLE row table
  is NVFP4 too; no arm in this document tells the two apart, and the 0.348 that survives the
  dense overlay is theirs to share with whatever the engine's own numerics contribute.
- **Engine numerics against quantization error.** The control arm proves that the BF16 dense
  path computes what the FP4 dense path computes (`docs/dense-overlay.md` 4.1); it does not
  prove that either is what the model should compute. A kernel that is subtly wrong in a way
  BOTH paths share is invisible to every arm here.
- **The llama arm's collection form.** The crow rows come from ONE prefill of the whole
  sequence; the llama rows come from 607 separate prefills of growing prefixes. Section 6
  sizes the difference between two llama collections at up to 0.05 mean KLD, and there is no
  way to give llama-server the crow form without a second tool.
- **Anything about German, long context or the sparse attention regime.** Both references are
  English code text at 298 and 607 tokens, where the QSA indexer is dense by construction.
  robin dropped German as a target on 2026-09-19; the long-context question stays open and
  this instrument does not reach it.
- **Perplexity, `Δp` on the corpus token, and the greedy-continuation agreement Unsloth
  publishes.** None of them is computed here.

## 8. How to re-run all of it

```
# the references (copy from the Windows tree once; about 900 MB)
#   decode_out/oracle-tf298/{ref-logits.f32,tf298-ids.json,ref-gen-sequence.json}
#   decode_out/oracle-t2b/{ref-logits.f32,t2b-tf-ids.json,ref-gen-sequence.json}

# one crow arm (27 to 100 s each; ONE engine on the card at a time)
systemd-run --user --scope --slice=session.slice --quiet -p MemorySwapMax=0 \
    -p MemoryHigh=56G -p MemoryMax=58G \
    env CROW_CNQ=converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq \
        CROW_HOTSETS=decode_out/hotsets-M-longctx2100-n160.json CROW_GRAPH=1 CROW_MMA=1 \
        LD_LIBRARY_PATH=$HOME/.local/share/crow/cuda/lib \
    engine/target/release/decode parity decode_out/oracle-t2b/t2b-tf-ids.json \
        decode_out/oracle-t2b/none
#   control / orig add CROW_CNQ_OVERLAY=$PWD/converter/dense-bf16-{control,originals}.cnq
#                  and CROW_PINNED_BUDGET_GB=50; kvbf16 adds CROW_KV=bf16; mma0 CROW_MMA=0

# the llama arm (7.6 min for 298 rows, 18.7 min for 607; the card must be free first)
~/.local/share/crow/venv/bin/python ~/.local/share/crow/tools/start-server.py flash-next-q2-k-xl &
tools/llama-row-probs.py --ids decode_out/oracle-t2b/t2b-tf-ids.json --round-trip
tools/llama-row-probs.py --ids decode_out/oracle-t2b/t2b-tf-ids.json --rows 0:607 \
    --no-cache-prompt --out decode_out/oracle-t2b/llama/gpu-logits.f32
kill -INT $(ps -C llama-server -o pid=)

# the reading (no GPU, 34 s for 298 rows x 7 arms, 69 s for 607 x 7)
tools/oracle-kld.py --ref decode_out/oracle-t2b/ref-logits.f32 --rows 0:607 \
    --prompt-rows 596 \
    --arm none=decode_out/oracle-t2b/none/gpu-logits.f32 \
    --arm orig=decode_out/oracle-t2b/orig/gpu-logits.f32 \
    --arm llama=decode_out/oracle-t2b/llama/gpu-logits.f32 \
    --paired orig,none --paired llama,none --json decode_out/oracle-t2b/kld-78.json

python tools/test_oracle_kld.py      # 55 tests, no GPU, no server, no reference file
```

There is deliberately NO gate item for any of this, for the reason `docs/dense-overlay.md` 5.6
gives: every arm needs files that are not in the repository and cannot be — a 105 GB container,
a 5.17 GB overlay, a 73 GB GGUF and 900 MB of reference logits. What the gate does carry is the
tool's own unit tests, which need none of them.
