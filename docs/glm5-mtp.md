# glm5_next MTP block (NextN, checkpoint layer 45)

crow-nest #182, open point O1 of [glm5-next-recipe.md](glm5-next-recipe.md). GLM-5.3-Flash ships one
multi-token-prediction block (`num_nextn_predict_layers` 1). HF transformers 5.16.1 drops it on load
(`_keys_to_ignore_on_load_unexpected = [r"layers\.45\.", r"layers\.\d+\.shared_head\."]`,
`modeling_glm5_next.py:1359`) and has no MTP forward, so there is no HF golden. This page fixes the
formula from the serving stacks that run this model, names where they differ, and describes the
reference (`oracle/glm5_mtp.py`) and the engine module (`engine/src/glm5_mtp.rs`).

MTP stays in the model (robin): it is a lossless lever, measured later as its own arm (plan step 23).

## 0. Sources

Read 2026-10-09 from raw.githubusercontent.com at these commits; nothing built or run.

| key | file | commit |
|---|---|---|
| llama.cpp | `src/models/glm5-next.cpp` (graph_mtp `:550-672`, trunk tail `:778-798`), `common/speculative.cpp` (draft-mtp `:1390-1800`) | `b9acf138a1e28ce1fc23b5a4fc4b12444b50f7ea` = merge of PR #29928 "feat: add GLM5Next MTP, optimize", 2026-10-07 |
| vLLM | `vllm/models/glm5next/common/mtp.py`, `…/common/model.py`, `…/nvidia/ops/fused_eh_norm.py`, `vllm/model_executor/models/deepseek_mtp.py`, `vllm/v1/spec_decode/llm_base_proposer.py` | `970dc63478da41409dec009690f462e8e33d22f2` (main) |
| SGLang | `python/sglang/srt/models/glm5_next_nextn.py`, `…/models/deepseek_nextn.py`, `…/layers/attention/index_topk_share.py` | `121aa8dcc72ff81db69b17c80bed2db5513e7ed4` (main) |
| paper | DeepSeek-V3 Technical Report, section 2.2, eq. 21-23 | arXiv 2412.19437v2 |
| HF | `transformers/models/glm5_next/modeling_glm5_next.py` | 5.16.1 (`.venv-oracle`) |

Line numbers below are `file:line` at these commits.

## 1. Formula of record

Per draft row `i` (trunk row `i` paired with the next token `t_{i+1}`; it drafts `t_{i+2}`):

| # | operation | sources |
|---|---|---|
| 1 | `h_i` = the trunk's lm_head input: `norm(mean over the 4 streams of layer 44's output)` (post final norm) | llama.cpp `glm5-next.cpp:784-794` ("the post-norm hidden state feeds the draft head"); vLLM `glm5next/common/model.py:854` (`self.norm(hidden_states)` is what the model returns) |
| 2 | `eh = [ enorm(embed(t_{i+1})) ∣ hnorm(h_i) ]`, both weighted RMSNorm eps 1e-5, **embedding first** | llama.cpp `glm5-next.cpp:598-604` (`ggml_concat(e_norm, h_norm, 0)`); vLLM `fused_eh_norm.py:39-49` (e at `[0, H)`, h at `[H, 2H)`); SGLang `deepseek_nextn.py:211-216` |
| 3 | `x = eh_proj · eh`, `eh_proj` [4096, 8192] BF16, no bias | llama.cpp `:183`, `:604`; vLLM `mtp.py:53`, `:108` |
| 4 | `y = x + MLA_DSA(input_layernorm(x))`: the block's own MLA latent cache and its own **full** indexer | llama.cpp `:606-614`, `:1066` ("the NextN block always has a full indexer"); vLLM `model.py:532-541` |
| 5 | `z = y + MoE(post_attention_layernorm(y))`: router + 288 routed experts + shared expert, as the trunk's MoE | llama.cpp `:621-652`; vLLM `model.py:541-551` |
| 6 | `n = shared_head.norm(z)`; `n` is both the lm_head input and the `h` of a chained next step | llama.cpp `:654-658`; vLLM `mtp.py:109-116`, `:193-196` |
| 7 | `logits = n · lm_head^T` with the **trunk's** lm_head (the checkpoint has no `shared_head.head`, no `embed_tokens` of the block) | llama.cpp `:665-668` (falls back to `model.output`); vLLM `llm_base_proposer.py:1571` (`sh.head = target_language_model.lm_head`) |

No mHC: the block has no `hc_*` tensors and runs as a plain pre-norm residual layer on one stream
(vLLM `model.py:531-532`, "70B or MTP layers: KDA + MoE without HC"; for an MTP layer that path builds MLA, `model.py:370`).

**Pairing and positions.** vLLM rotates the token ids left by one and keeps the positions
(`llm_base_proposer.py:842-850`): row `i` = (`t_{i+1}`, `h_i`) at position `i`. SGLang inherits the same
EAGLE shift (`glm5_next_nextn.py:24` → `deepseek_nextn.py`). With NoPE the position only orders the
block's cache and aligns the indexer's 4-token pools.

**`index_share_for_mtp_iteration`** (config.json: `true`). With it, draft step 0 runs the block's own
indexer and steps 1+ of the same draft reuse step 0's top-k instead of running the indexer again
(vLLM `mtp.py:151-154`, `llm_base_proposer.py:578-582`, `:607-613`, `:1592-1603`; SGLang
`index_topk_share.py:24-30`, `:77-97`). It never shares the trunk's top-k. With one draft step per
verify (this reference, the golden) it changes nothing.

**DeepSeek-V3 paper** (arXiv 2412.19437v2, section 2.2, eq. 21-23): `h'_i = M_k [RMSNorm(h_i^{k-1});
RMSNorm(Emb(t_{i+k}))]`, `h_k = TRM_k(h')`, `P_{i+k+1} = OutHead(h_k)`, embedding and output head shared
with the main model. The written concat order is **h first**; every code base above puts the
embedding first (vLLM `deepseek_mtp.py:127-128` for DeepSeek itself). Section 4 measures it: h first gives
0 / 89 agreement on the GLM weights.

## 2. Differences between the sources

| # | topic | SGLang | vLLM | llama.cpp (PR #29928) | reference |
|---|---|---|---|---|---|
| M1 | row 0 | (`t_1`, `h_0`), nothing masked | embedding zeroed where position == 0 (`deepseek_mtp.py:122-123`, `fused_eh_norm.py:41`) | a leading row (`t_0`, h = 0) at position 0, then (`t_{i+1}`, `h_i`) at `i + 1` (`speculative.cpp:1412`, `:1494`, `:1575-1590`) | SGLang is the primary; the other two run as variants (section 4) |
| M2 | concat order | e first | e first | e first | e first; the paper's h-first runs as a variant |
| M3 | final residual + norm | add, then `shared_head.norm` | fused add + norm (`mtp.py:115`) | add, then norm | add, then norm (same math) |
| M4 | recycled h for step 2 | post `shared_head.norm` | post-norm (`mtp.py:109-111`, `deepseek_mtp.py:139-144`) | post-norm (`:656-658`) | not exercised (one step) |

## 3. Code

- `oracle/glm5_mtp.py`: `Glm5MtpBlock` is HF's `Glm5NextTextAttention` and `Glm5NextTextMoE` plus
  `enorm`, `hnorm`, `eh_proj`, the two layernorms and `shared_head.norm`, its state-dict keys equal to
  the checkpoint names under `layers.45.` (25 tensors + 3 x 288 experts; `test_glm5_mtp.Names` checks
  them against the real index). `mtp_text_config` adds the block to the layer lists (DSA, sparse,
  full indexer) so a `DynamicCache` holds its slot. `mtp_forward` runs the runner's call plan (prompt
  calls, then one row per call) against the block's own cache. `MixedSource` reads the experts from
  the 3-bit container and every other tensor of layer 45 from the FP8 originals (section 5). `run`
  reads a trunk dir written with `--capture-head` (#165: `head-norm.f32`, `head-logits.f32`).
- `engine/src/glm5_mtp.rs`: `MtpPass::call` = the fused `mtp_eh_norm` kernel (zero at position 0
  optional + two RMSNorms + concat), `eh_proj` on `gm_gemm`, `glm5_mla::MlaScratch::forward` /
  `forward_with` (codec hook) against the block's own `MlaCache`, `glm5_moe::GpuMoePlan::run` over the
  block's MUL1 records, `mtp_add` residuals, `shared_head.norm` on `gm_rmsnorm`; result `normed`, for
  `glm5_head::Head::lm_head` with the trunk's lm_head. `pair_rows` (with `Pos0` for M1),
  `mtp_tensors` (the plan of layer 45, FP8 split, overlay codec), `missing_in`, `overlay_bytes`,
  `load_mtp_records` (the container's `mtp` section into one VRAM buffer + `[E]` table),
  `mtp_overlay_check` + `load_mtp` (container + MTP overlay -> `MtpBlock`, the NVFP4 projections on
  `gemv_fp4_b`), `MtpPass::call_tapped` / `MtpTaps` (the golden's intermediate rows).

## 4. Golden and evidence

Oracle selftest (synthetic 8-layer config, small shapes, an MTP layer appended in the original naming
and FP8 format, trunk from HF's full model): the oracle (HF blocks, prompt in calls of 7 rows, decode
rows singly) equals a plain-torch formula without any HF module (`manual_mtp`) to 2.3e-6 on the head
norm and 1.2e-6 on the logits over 27 rows. The proof goes red with the paper's order, with the
pre-norm trunk state, and with a block cache that forgets its history (`test_glm5_mtp.py`, 10 tests).

Engine: `glm5_mtp_gpu_block_matches_the_host_composition` (RTX 5090, real shapes, synthetic weights,
3 prompt rows then 3 decode rows): `MtpPass` against the host composition of the blocks' own
references (glue twin, `glm5_mla::host::forward`, `glm5_moe::moe_cpu`): 1 − cosine ≤ 2.4e-8 per row,
routing sets identical; red (1 − cosine 0.997) with the concat order swapped in the kernel.

Real weights (2026-10-09, `runs/glm53-flash/step06/README.md` "MTP golden", record `golden-mtp.json`): the
90 ids of step 6 over the full 45-layer trunk on the 3-bit container (`ref-mul1-all`, #165 `--capture-head`), the
block's experts from the container, its other 25 tensors from the FP8 originals. Golden
`models/GLM-5.3-Flash-step06/ref-mul1-mtp/`, 14 files, manifest sha256 `eaffa03a…ffc13d`, 89 s CPU, RSS 28.3 GiB.
The engine's glue twin on the checkpoint's BF16 `enorm` / `hnorm` / `eh_proj` equals the oracle's `mtp-eh` to
2.8e-6 absolute (rel RMS 3.6e-7, `glm5_mtp_glue_matches_the_oracle_golden`).

Draft top-1 equal to the trunk's own top-1 for the same id, 89 rows, one step (an acceptance indication, not a gate):

| pairing | agreement |
|---|---|
| SGLang (primary) | 55 / 89 = 0.618 |
| vLLM (row 0 embedding zeroed) | 57 / 89 = 0.640 |
| llama.cpp (leading `(t_0, 0)` row) | 56 / 89 = 0.629 |
| paper order `[h; e]` | **0 / 89** |
| trunk state before the final norm | 52 / 89 = 0.584 |

The paper's order gives no agreement: the weights were trained embedding-first (M2 settled by measurement). The
three stacks' row-0 rules differ by 1-2 rows of 89, within noise at this size. 4 decode rows only (0 / 4 in every
variant); the 61.8 % is a first indication on one short text with uncalibrated (identity-Hessian) experts and FP8
attention, not the engine's acceptance.

## 5. Weights: what the 3-bit container holds

`GLM-5.3-Flash-MUL1K3.cnq` holds only the block's 288 routed experts (section `mtp`, MUL1 K=3, 864
index entries); the `cnq4.5-glm5-next` row omits the rest of layer 45 and `--experts-mul1` adds only
its experts (`docs/glm-mul1-conversion.md` "Container", `converter/src/recipe.rs`
`mul1_expert_decision`). The other 25 tensors (7 of them FP8 in the checkpoint: q_a, q_b, kv_a, o_proj,
shared gate/up/down) are not in any engine-readable file. The oracle reads them from the FP8 originals;
the engine needs an overlay. Sizes (`glm5_mtp::overlay_bytes`, derived):

| overlay | bytes | note |
|---|---|---|
| trunk codecs (NVFP4 for q_a/q_b/kv_a/kv_b/o_proj and the shared expert, BF16 for eh_proj, norms, indexer, router; F32 score bias) | 164,674,208 B (157.0 MiB) | the decisions the trunk's DSA + MoE layers carry; `glm5_mtp::Store`; eh_proj BF16 (llama.cpp keeps `nextn.eh_proj` at Q8_0 or higher, recipe D10) |
| all BF16 | 369,670,784 B (352.5 MiB) | `MlaScratch::forward` runs it without a hook |

Built 2026-10-09 (#182): the trunk-codec overlay as a separate index v2 file,
`converter --mtp-overlay` -> `converter/GLM-5.3-Flash-MTP-overlay.cnq` (164,771,937 B, payload 164,674,176 B,
sha256 `191d5e18…da962a`; `docs/glm-mul1-conversion.md` "MTP overlay"). `load_mtp(base, overlay)` checks it
against the container (`mtp_overlay_check`) and loads the whole block; `MtpBlock::call` runs the NVFP4 projections.

On the GPU against the golden (which used the FP8 originals for these 25 tensors, so not bit-equal), 89 rows:
eh cosine 1.000000; head norm cosine mean 0.98395, min 0.96957; routing overlap 0.926; DSA selection 89 / 89
identical; draft top-1 = golden draft on 79 / 89; draft = trunk's next on 57 / 89 (golden 55 / 89).

## 6. Integration (#192)

Built: the speculative decode `CROW_GLM_MTP=N` in `Glm5Run::generate` ([glm5-model.md](glm5-model.md)
section 6.4: verify call, KDA rollback, counters, tests).

- After each step the block runs once over the accepted rows with `h` = the verify's `normed` rows
  and `e` = the embeddings of the accepted ids (the prompt likewise, in calls of `MTP_CHUNK` = 16
  rows); its last row drafts. Chained drafts (N > 1) feed the block's own `normed` back as `h`;
  `index_share_for_mtp_iteration` is not built (each chained row runs its own indexer: the selection
  buffers are `MlaScratch`'s and `glm5_mla` has no "skip the indexer" entry). It changes drafts only.
- The weights: `load_mtp(base, overlay)` -> `MtpBlock` (`Glm5Run::mtp_from_env`); one draft =
  `MtpBlock::call` then `glm5_head::Head::lm_head` + argmax with the trunk's lm_head. The block's four
  NVFP4 projections run on `glm5_gemv_fp4` (#191), bit-identical to `gemv_fp4_b`
  (`glm5_mtp_spec_gpu_block_gemv_is_the_record_kernel`).
- The block's 288 records sit in VRAM (2,728,525,824 B), outside the tiers; `glm5_run` plans the
  tiers at free VRAM minus `spec_vram_bytes`.

## 7. The MTP experts' Hessian (not done here)

The 288 MTP records were quantized with an identity Hessian because no MTP forward existed
(`docs/glm-mul1-conversion.md`). With this reference the real one is: per calibration file, the trunk head
`norm(mean(l44-output))` (the four `decode_out/glm-mul1/work/<file>/l44-output.bf16` FP8-trunk states are on disk,
`progress.json` `layer_done` 44), the pairing of section 1, the block forward on the FP8 originals, a pre-hook on
the block's `mlp` (the `capture` hook) for `X` and the routing; then `quantize` the 288 experts with
`H_gu = X^T X` and `H_down` as for layers 3-44, and replace the 288 records of section `mtp` (same size,
9,474,048 B each). Derived cost: block forward about 96 s per file (one DSA + MoE layer,
`docs/glm-mul1-conversion.md` capture row) x 4 = 6.4 min plus a 7.8 s FP8 layer load; quantize 288 x 1.04-1.31 s
= 5.0-6.3 min (RTX 5090). The converter has no path that swaps one section's records in place; a full rebuild
from the store is the alternative. Whether it is worth it is the MTP arm's measurement (plan step 23).
