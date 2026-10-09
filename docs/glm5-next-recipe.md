# glm5_next layer recipe (GLM-5.3-Flash)

Step 2 of the GLM-5.3-Flash plan (crow-nest #153, parent Crow #362). This page is the single
written description of what one forward pass of `glm5_next` computes: per building block the
checkpoint tensors, shapes, dtypes, norms, clamps, caches and the order of operations. The
converter recipe (steps 4/5), the layerwise reference runner (step 7), the planner (step 11) and
the kernels (steps 13a-e) take their numbers from here instead of re-deriving the model.

Nothing on this page is measured. Every rule is read from source; derived numbers carry their
formula.

## 0. Sources and citation keys

| key | file | version |
|---|---|---|
| `M:n` | `transformers/models/glm5_next/modeling_glm5_next.py` line n (2385 lines) | transformers 5.16.1, `crow-nest/.venv-oracle` |
| `C:n` | `transformers/models/glm5_next/configuration_glm5_next.py` (319 lines) | same |
| `CM:n` | `transformers/conversion_mapping.py` | same |
| `CU:n` | `transformers/cache_utils.py` | same |
| `FP8:n` | `transformers/integrations/finegrained_fp8.py` | same |
| `cfg` | `config.json` of `zai-org/GLM-5.3-Flash` rev `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a`, 69,416 B, sha256 `bb8f01c4…3e9f` (read 2026-10-08) | |
| `idx` | `model.safetensors.index.json` of the same rev, 8,406,613 B, `total_size` 328,326,771,576 | |
| `g5:n` | llama.cpp `src/models/glm5-next.cpp` | merge `649dcb1036fd47d01a83b56c5e35913e65c277ba` (PR #27773) |
| `glm.py:n`, `graph:n`, `hyb:n`, `dn:n`, `lm:n` | llama.cpp `conversion/glm.py`, `src/llama-graph.cpp`, `src/llama-memory-hybrid-idx.cpp`, `src/models/delta-net-base.cpp`, `src/llama-model.cpp` | same merge |

The llama.cpp commit is in no local tree (`git cat-file -e` fails in `crow-lab/src`, every
`crow-lab/wt-*` and `llama-up`); the files were read from `raw.githubusercontent.com` at the merge
commit, never built or run.

**Rule of record:** the HF file is the reference. Where llama.cpp differs, HF applies and the
difference stands in section 12 and as a comment on #153.

## 1. Shape of the model (cfg)

| quantity | value | read by HF at |
|---|---|---|
| `hidden_size` H | 4096 | `M:1417`, everywhere |
| `vocab_size` | 154,880 | `M:1417`, `M:2075` |
| layers (trunk) | 45 (`num_hidden_layers`); layer 45 in the checkpoint is the MTP block (section 11) | `M:1418-1420` |
| `layer_types` | `linear_attention` (KDA) at 34 layers {0,1,2,4,5,6,…,44} = every i with i mod 4 != 3; `deepseek_sparse_attention` (MLA+DSA) at 11 layers {3,7,11,…,43} | `M:1262-1268` |
| `mlp_layer_types` | `dense` 0-2, `sparse` (MoE) 3-44 | `M:1270-1272` |
| `indexer_types` | all 45 `"full"` → every DSA layer runs its own indexer; the cross-layer sharing path is inactive | `M:1130-1134` |
| `hc_mult` | 4 residual streams | `M:254`, `M:1477` |
| `rms_norm_eps` | 1e-5 (all RMSNorms incl. the mHC input norm and the KDA gated norm) | `M:1274-1275`, `M:257`, `M:601`, `M:622` |
| `hc_eps` | 1e-6 | `M:256` |
| `hc_sinkhorn_iters` | 20 | `M:255` |
| `swiglu_limit` | 10.0 (dense MLP, shared expert, routed experts) | `M:96`, `M:118` |
| position encoding | none (NoPE): `qk_rope_head_dim` 0, `position_embeddings=None` | `M:1485-1486`, `C:225-228` |
| `tie_word_embeddings` | false; `lm_head` is its own tensor | `M:2075` |
| `eos_token_id` | [154820, 154827, 154829]; `pad_token_id` 154820 | `C:126`, cfg |

`index_kpool` is **4** in cfg; the HF class default is 16 (`C:153`), so any code that builds the
config without the checkpoint's `config.json` gets the wrong pool size.

## 2. Checkpoint conventions: names, FP8, dtypes

- **Names.** Checkpoint keys are `model.language_model.layers.<L>.<…>`, `model.language_model.embed_tokens.weight`,
  `model.language_model.norm.weight`, `lm_head.weight` (idx). HF renames on load (`CM:535-580`):
  `self_attn.{f_a_proj,f_b_proj,dt_bias,A_log}` → `self_attn.forget_gate.*`; `hc_attn_{fn,base,scale}` → `attn_hc.{fn,base,scale}`;
  `hc_ffn_*` → `ffn_hc.*`; per-expert `mlp.experts.<e>.{gate,up}_proj.weight` → one `mlp.experts.gate_up_proj [E, 2I, H]`
  (MergeModulelist dim 0, then Concatenate dim 1: **gate rows first, then up**, `CM:558-565`, used as `gate, up = chunk(2)` at `M:138`);
  `mlp.experts.<e>.down_proj.weight` → `mlp.experts.down_proj [E, H, I]` (`CM:566-570`);
  `self_attn.{q,k,v}_conv1d.weight` → one `self_attn.conv1d.weight`, concatenated along dim 0 in the order q, k, v (`CM:571-579`).
- **Tensor count.** 76,108 tensors in idx: 347 vision (out of scope, step 20), 38,423 text tensors plus 37,338
  `weight_scale_inv` companions. The 38,423 text names match the shapes derived on this page one for one
  (checked by script against idx, 2026-10-08).
- **FP8.** `quantization_config`: `quant_method` fp8, `fmt` e4m3, `activation_scheme` dynamic, `weight_block_size` [128, 128],
  1,509 `modules_to_not_convert` entries. A tensor is FP8 **iff** idx has a `<name>_scale_inv` companion; that set equals
  "not in `modules_to_not_convert`" for every text tensor (script check: 0 mismatches of 38,423). Dequant (`FP8:1003-1048`): the block grid is taken from the scale shape
  ([ceil(rows/128), ceil(cols/128)]; every FP8 shape below divides by 128), `w = fp8.to(f32) * scale_inv` per 128×128 block, in f32,
  then cast to the model dtype; a non-divisible shape is refused (`FP8:1023-1026`).
- **Non-FP8 tensors** are written "BF16*" below: cfg `dtype` bfloat16, but the exact stored dtype of the small ones (A_log, dt_bias,
  e_score_correction_bias, `hc_*`, conv, norms) is not in idx (open point O2). HF keeps
  `e_score_correction_bias`, `conv1d`, `dt_bias`, `A_log` in f32 at run time (`M:1358`).
- **Byte balance** (derived): FP8 values 314,396,639,232 B + 19,189,248 scales × 4 B + 6,362,765,150 non-FP8 values × 2 B
  = 327,198,926,524 B; idx `total_size` minus that = 1,127,845,052 B for the 347 vision tensors and any text tensor stored wider than BF16.

## 3. Forward pass, top level

| # | operation | HF |
|---|---|---|
| 1 | `e = embed_tokens[ids]`, [154880, 4096] BF16* (not FP8) | `M:1447-1448` |
| 2 | `X = e` broadcast to 4 identical streams, `X ∈ [T, 4, 4096]`, model dtype (BF16) | `M:1477` |
| 3 | for L in 0..44: `X = DecoderLayer_L(X)` (section 4) | `M:1480-1491` |
| 4 | `h = mean over the 4 streams` (unweighted, "unlike DeepSeek-V4") | `M:298-302`, `M:1493` |
| 5 | `h = norm(h)`, RMSNorm weight [4096] BF16*, eps 1e-5 | `M:1421`, `M:1493` |
| 6 | `logits = lm_head(h)`, [154880, 4096] BF16* (not FP8, untied) | `M:2075`, `M:2179-2181` |

The MTP block (layer 45) is not part of this pass in HF (section 11).

## 4. Decoder layer (all 45 layers)

Weighted RMSNorm (`M:66-83`): `x32 = x.to(f32)`; `x32 * rsqrt(mean(x32²) + eps)`; **cast back to the input dtype, then multiply by
the weight** (`M:80`). Unweighted RMSNorm for mHC (`M:210-216`): `x * rsqrt(mean(x.float()²) + eps)`, no weight.

| # | operation | HF |
|---|---|---|
| 1 | `R = X` (residual, [T, 4, 4096]) | `M:1293` |
| 2 | `post, comb, h = attn_hc(X)` (section 5) | `M:1294` |
| 3 | `h = input_layernorm(h)`, weight [4096] BF16*, eps 1e-5 | `M:1274`, `M:1296` |
| 4 | `a = KDA(h)` (section 6) or `a = MLA_DSA(h)` (sections 7-8), [T, 4096] | `M:1298-1315` |
| 5 | `X[i] = post[i]·a + Σ_j comb[j,i]·R[j]` for each stream i; `post` and `comb` are **cast to BF16 first**, the mix runs in BF16 | `M:1316-1318` |
| 6 | `R = X`; `post, comb, h = ffn_hc(X)` | `M:1320-1321` |
| 7 | `h = post_attention_layernorm(h)`, weight [4096] BF16*, eps 1e-5 | `M:1275`, `M:1323` |
| 8 | `f = MLP(h)` (dense, section 10.1) or `f = MoE(h)` (sections 9-10) | `M:1270-1272`, `M:1324` |
| 9 | `X[i] = post[i]·f + Σ_j comb[j,i]·R[j]`, BF16 as in step 5 | `M:1325-1327` |

## 5. mHC (manifold-constrained hyper-connections), two per layer

Tensors per layer L in 0..44, site s ∈ {attn, ffn} (idx; HF module `attn_hc` / `ffn_hc`, `M:1277-1278`):

| checkpoint tensor | shape | dtype | HF |
|---|---|---|---|
| `layers.L.hc_s_fn` | [24, 16384] = [(2+4)·4, 4·4096] | BF16* | `M:258-259` |
| `layers.L.hc_s_base` | [24] | BF16* | `M:260` |
| `layers.L.hc_s_scale` | [3] (pre, post, comb) | BF16* | `M:265` |

Order of operations (`M:267-295`), all in f32 unless stated:

| # | operation | HF |
|---|---|---|
| 1 | `flat = X.flatten(4·4096).float()`, unweighted RMSNorm over all 16,384 values, eps **`rms_norm_eps` 1e-5** (not `hc_eps`) | `M:257`, `M:278` |
| 2 | `m = flat @ fn.float()ᵀ` → 24 values; split [4 pre, 4 post, 16 comb] | `M:279` |
| 3 | `pre = sigmoid(m_pre·scale[0] + base[0:4]) + 1e-6` | `M:283` |
| 4 | `post = 2·sigmoid(m_post·scale[1] + base[4:8])` (range [0, 2]) | `M:284` |
| 5 | `comb_logits = m_comb.view(4,4)·scale[2] + base[8:24].view(4,4)`; element [j,i]: row j = source stream, column i = destination stream (use in `M:1316-1318`) | `M:285` |
| 6 | `comb = softmax over the last dim (i) + 1e-6` | `M:286` |
| 7 | `comb = comb / (Σ_j comb + 1e-6)` (column normalise, sum over dim −2) | `M:287` |
| 8 | 19 times (`hc_sinkhorn_iters − 1`): `comb /= (Σ_i comb + 1e-6)` (rows), then `comb /= (Σ_j comb + 1e-6)` (columns) | `M:288-290` |
| 9 | `h = Σ_j pre[j]·X[j]` in f32, cast to the model dtype | `M:294` |

`post` and `comb` leave the block in f32 and are cast to BF16 by the decoder layer (section 4 steps 5/9).

## 6. KDA (Kimi delta attention) — 34 layers

Tensors per KDA layer (idx; all non-FP8: every KDA tensor is in `modules_to_not_convert`, incl. `o_proj`):

| checkpoint tensor | HF module | shape | dtype | HF |
|---|---|---|---|---|
| `self_attn.q_proj.weight` | `q_proj` | [8192, 4096] (64 heads × 128) | BF16* | `M:603` |
| `self_attn.k_proj.weight` | `k_proj` | [8192, 4096] | BF16* | `M:604` |
| `self_attn.v_proj.weight` | `v_proj` | [8192, 4096] | BF16* | `M:605` |
| `self_attn.{q,k,v}_conv1d.weight` | `conv1d.weight` (concat q,k,v) | 3 × [8192, 1, 4] → [24576, 1, 4], depthwise, no bias | BF16*, held f32 | `M:607-615`, `M:1358`, `CM:571-579` |
| `self_attn.f_a_proj.weight` | `forget_gate.f_a_proj` | [128, 4096] | BF16* | `M:312` |
| `self_attn.f_b_proj.weight` | `forget_gate.f_b_proj` | [8192, 128] | BF16* | `M:313` |
| `self_attn.dt_bias` | `forget_gate.dt_bias` | [8192] | BF16*, held f32 | `M:314` |
| `self_attn.A_log` | `forget_gate.A_log` | [64] | BF16*, held f32 | `M:315` |
| `self_attn.b_proj.weight` | `b_proj` | [64, 4096] | BF16* | `M:618` |
| `self_attn.g_a_proj.weight` | `g_a_proj` | [128, 4096] | BF16* | `M:620` |
| `self_attn.g_b_proj.weight` | `g_b_proj` | [8192, 128] | BF16* | `M:621` |
| `self_attn.o_norm.weight` | `o_norm` | [128] (per head) | BF16* | `M:622` |
| `self_attn.o_proj.weight` | `o_proj` | [4096, 8192] | BF16* | `M:623` |

Order of operations (`M:628-733`), input `x = h` from section 4 step 3:

| # | operation | HF |
|---|---|---|
| 1 | zero padded positions | `M:636` |
| 2 | `qkv = cat(q_proj x, k_proj x, v_proj x)` [24576] | `M:642-649` |
| 3 | causal depthwise conv over time, kernel 4, no bias, then **SiLU**; computed in the conv weight's dtype (f32), result cast back | prefill `M:394-413`, decode `M:374-390`, call `M:658-683` |
| 4 | split q, k, v, each [64, 128] | `M:685-693` |
| 5 | forget gate: `g_raw = f_b(f_a(x)).float() + dt_bias` [64, 128]; `g = −5 · sigmoid(exp(A_log[h]) · g_raw)` — the `gate_lower_bound` −5 path (cfg sets it, so the softplus branch `M:331-335` is dead) | `M:319-329`, `C:195` |
| 6 | `beta = sigmoid(b_proj x)` [64] | `M:697` |
| 7 | in f32: `q = l2norm(q)`, `k = l2norm(k)` with `x / sqrt(Σx² + 1e-6)`; `q *= 128^-0.5` | `M:416-424`, `M:441-452` (decode), `M:497-514` (prefill) |
| 8 | per head, state `S ∈ [128 (key), 128 (value)]` f32: `S ← diag(exp g_t) · S` (decay per key channel); `u = beta_t · (v_t − Sᵀk_t)`; `S ← S + k_t uᵀ`; `o_t = Sᵀ q_t` | decode `M:464-476`; prefill is the same recurrence in chunks of 64 `M:482-578` |
| 9 | `o` cast to the model dtype; state stored as f32 | `M:478`, `M:725-726` |
| 10 | `gate = g_b(g_a(x))` [64, 128] | `M:729` |
| 11 | gated RMSNorm per head over 128: f32, `rsqrt(mean(o²) + 1e-5)`, × `o_norm.weight` (f32), × `sigmoid(gate)`, cast back | `M:346-358`, `M:730` |
| 12 | `a = o_proj(o)` [4096] | `M:731` |

State per sequence (section 13): recurrent `S` 64 × 128 × 128 f32; conv window of the last pre-conv `qkv` rows.

## 7. MLA attention without RoPE — 11 trunk layers (+ MTP)

Tensors per DSA layer (idx):

| checkpoint tensor | HF module | shape | dtype | HF |
|---|---|---|---|---|
| `self_attn.q_a_proj.weight` | `q_a_proj` | [1536, 4096] | FP8, scale [12, 32] | `M:1097-1101` |
| `self_attn.q_a_layernorm.weight` | `q_a_layernorm` | [1536], RMSNorm eps 1e-5 | BF16* | `M:1102-1104` |
| `self_attn.q_b_proj.weight` | `q_b_proj` | [16384, 1536] = [64 × 256, 1536] | FP8, scale [128, 12] | `M:1105-1109` |
| `self_attn.kv_a_proj_with_mqa.weight` | `kv_a_proj_with_mqa` | [512, 4096] (512 latent + 0 rope) | FP8, scale [4, 32] | `M:1111-1115` |
| `self_attn.kv_a_layernorm.weight` | `kv_a_layernorm` | [512], RMSNorm eps 1e-5 | BF16* | `M:1116` |
| `self_attn.kv_b_proj.weight` | `kv_b_proj` | [32768, 512] = [64 × (256 k + 256 v), 512] | BF16* | `M:1117-1121` |
| `self_attn.o_proj.weight` | `o_proj` | [4096, 16384] = [4096, 64 × 256] | FP8, scale [32, 128] | `M:1123-1127` |

No bias anywhere (`attention_bias` false). Order of operations (`M:1155-1216`), input `x = h`:

| # | operation | HF |
|---|---|---|
| 1 | `q_resid = q_a_layernorm(q_a_proj x)` [1536] (also feeds the indexer) | `M:1167` |
| 2 | `q = q_b_proj(q_resid)` → [64, 256] | `M:1168` |
| 3 | `c = kv_a_layernorm(kv_a_proj_with_mqa x)` [512] (the rope part is empty) | `M:1170-1173` |
| 4 | `kv = kv_b_proj(c)` → [64, 512]; per head rows `[h·512, h·512+256)` = `k_nope`, `[h·512+256, (h+1)·512)` = `v` | `M:1142-1146` |
| 5 | `K = k_nope` (k_rot has width 0) | `M:1147-1153` |
| 6 | HF caches the **expanded** K and V (section 13) | `M:1178-1179` |
| 7 | `topk = indexer(x, q_resid)` (section 8) | `M:1181-1187` |
| 8 | mask: a key is visible iff its index is in `topk` (causality and padding are already inside `topk`) | `M:1193-1197`, `M:1218-1256` |
| 9 | `p = softmax(q·Kᵀ · 256^-0.5 + mask)` in f32, cast to the model dtype; `o = p·V` | `M:1128`, `M:1052-1058` |
| 10 | `a = o_proj(o.reshape(64·256))` | `M:1214-1215` |

**Latent (absorbed) form** — derived, not HF code; it is algebraically equal to steps 4-9 and is what llama.cpp runs (`g5:941-945`, `g5:982-984`):
with `W_k,h = kv_b_proj.weight[h·512 : h·512+256, :]` and `W_v,h = kv_b_proj.weight[h·512+256 : (h+1)·512, :]` (each [256, 512]):
`q̃_h = q_h · W_k,h` [512]; `score_t,h = q̃_h · c_t · 256^-0.5`; `o_h = (Σ_t p_t,h c_t) · W_v,hᵀ` [256].
The cache then holds `c_t` only (the normalised latent, 512 values). The golden for step 7 is HF's expanded order.

## 8. DSA indexer with k-pool 4, `index_topk` 2048 — every DSA layer

Tensors per DSA layer (idx; all non-FP8):

| checkpoint tensor | shape | dtype | HF |
|---|---|---|---|
| `self_attn.indexer.wq_b.weight` | [4096, 1536] = [32 heads × 128, q_lora 1536] | BF16* | `M:761` |
| `self_attn.indexer.wk.weight` | [128, 4096] | BF16* | `M:762` |
| `self_attn.indexer.k_norm.weight`, `.bias` | [128], [128] — **LayerNorm** (mean-centred, with bias), eps 1e-6 | BF16* | `M:763` |
| `self_attn.indexer.weights_proj.weight` | [32, 4096] | BF16* | `M:764` |
| `self_attn.indexer.index_kpool_compress_gate` | [128, 4096] | BF16* | `M:771` |
| `self_attn.indexer.index_kpool_compress_ape` | [4, 128] (pool position × channel) | BF16* | `M:770` |

Order of operations (`M:773-877`), inputs `x` (section 4 step 3) and `q_resid` (section 7 step 1); `@torch.no_grad` (`M:773`):

| # | operation | HF |
|---|---|---|
| 1 | `iq = wq_b(q_resid)` → [32, 128] — **no RoPE** | `M:797` |
| 2 | `ik = LayerNorm(wk x)` [128] — **no RoPE** | `M:798` |
| 3 | `ig = x @ compress_gateᵀ` [128] (per-token, per-channel pool logits) | `M:800` |
| 4 | cache row per token: `[ik 128 | ig 128 | valid 1]` = 257 values, model dtype; appended to the indexer cache | `M:801-810`, `CU:338-351` |
| 5 | pools: consecutive groups of 4 tokens starting at the sequence's first valid token (position 0 for an unpadded sequence); a pool is valid only if all 4 members are valid | `M:899-959` |
| 6 | pooled key per pool: `pk[c] = Σ_{j<4} softmax_j(ig_j[c] + ape[j, c]) · ik_j[c]` — softmax over the 4 positions per channel in f32, **probabilities cast to the key dtype (BF16), product and sum in BF16** | `M:961-967` |
| 7 | `s_h = relu(iq_h · pk · 128^-0.5)` in f32 for every pool | `M:825-826` |
| 8 | `w = weights_proj(x).float() · 32^-0.5`; `score = Σ_h w_h · s_h` | `M:829-830` |
| 9 | candidate = valid pool whose **last** token is causally visible to the query; others get `finfo.min` | `M:832-844`, `M:879-897` |
| 10 | select `min(2048 / 4, n_pools) = ≤ 512` pools by score (top-k, unordered) | `M:847-852` |
| 11 | expand selected pools to their 4 raw token indices (≤ 2048 indices); invalid → −1 | `M:853-864` |
| 12 | append the incomplete tail as raw indices: the `visible_count mod 4` newest visible tokens (≤ 3) | `M:866-869`, `M:974-1024` |
| 13 | output width `2048 + 3 = 2051` per query, padded with −1, int32 | `M:871-877` |

Consequences (derived): a query at position p (0-based, unpadded) has `⌊(p+1)/4⌋` complete visible pools and `(p+1) mod 4` tail
tokens. While `⌊(p+1)/4⌋ ≤ 512`, i.e. for p + 1 < 2052, every pool is selected and the layer equals dense causal attention; from
p + 1 = 2052 on, each query attends to at most 2051 tokens. Scoring cost per query per DSA layer is linear in the number of pools
(32 heads × 128 × ⌊(p+1)/4⌋ MACs).

## 9. Router (MoE layers 3-44, MTP)

| checkpoint tensor | shape | dtype | HF |
|---|---|---|---|
| `mlp.gate.weight` | [288, 4096] | BF16* | `M:151` |
| `mlp.gate.e_score_correction_bias` | [288] | BF16*, held f32 | `M:156`, `M:1358` |

Order of operations (`M:158-183`), input `x` = output of `post_attention_layernorm`:

| # | operation | HF |
|---|---|---|
| 1 | `logits = x.float() @ gate.weight.float()ᵀ` [288] — f32 | `M:160` |
| 2 | `s = sigmoid(logits)` | `M:161` |
| 3 | `s_choice = s + e_score_correction_bias` (used for **choice only**) | `M:162` |
| 4 | group step with `n_group` 1, `topk_group` 1 (top-2 sum per group, keep 1 of 1 group) — a no-op here | `M:163-176` |
| 5 | `idx = top-8 of s_choice` (unordered) | `M:177` |
| 6 | `w = s[idx]` (uncorrected scores) | `M:178` |
| 7 | `w = w / (Σ w + 1e-20)` (`norm_topk_prob` true) | `M:179-181` |
| 8 | `w = w · 2.5` (`routed_scaling_factor`) | `M:182` |

## 10. Experts, shared expert, dense layers

SwiGLU with clamp, identical rule in all three (`M:98-104`, `M:137-142`): `gate = min(gate, 10)`, `up = clamp(up, −10, 10)`,
`y = down(silu(gate) · up)`. Only the gate's upper side is clamped.

### 10.1 Dense MLP, layers 0-2

| checkpoint tensor | shape | dtype | HF |
|---|---|---|---|
| `mlp.gate_proj.weight` | [12288, 4096] | FP8, scale [96, 32] | `M:92` |
| `mlp.up_proj.weight` | [12288, 4096] | FP8, scale [96, 32] | `M:93` |
| `mlp.down_proj.weight` | [4096, 12288] | FP8, scale [32, 96] | `M:94` |

### 10.2 Routed experts (288 per MoE layer) and shared expert

| checkpoint tensor | shape | dtype | HF |
|---|---|---|---|
| `mlp.experts.<e>.gate_proj.weight` | [2048, 4096] | FP8, scale [16, 32] | fused into `gate_up_proj[e, 0:2048]` `M:116`, `CM:558-565` |
| `mlp.experts.<e>.up_proj.weight` | [2048, 4096] | FP8, scale [16, 32] | `gate_up_proj[e, 2048:4096]` `M:116` |
| `mlp.experts.<e>.down_proj.weight` | [4096, 2048] | FP8, scale [32, 16] | `down_proj[e]` `M:117`, `CM:566-570` |
| `mlp.shared_experts.{gate,up}_proj.weight` | [2048, 4096] (`moe_intermediate_size` × `n_shared_experts` 1) | FP8, scale [16, 32] | `M:196-198` |
| `mlp.shared_experts.down_proj.weight` | [4096, 2048] | FP8, scale [32, 16] | `M:196-198` |

MoE order (`M:200-207`): router on `x` → `y_e = SwiGLU_e(x) · w_e` summed over the 8 selected experts (`M:131-134`) →
`+ shared_expert(x)` on the same input `x` (`M:206`). The routed sum accumulates in the model dtype (`M:134`).

## 11. MTP block (checkpoint layer 45)

Present in idx (shard 1 and others), 1,760 tensors:

| checkpoint tensor | shape | dtype | HF |
|---|---|---|---|
| `layers.45.enorm.weight`, `layers.45.hnorm.weight` | [4096] each | BF16* | no module; dropped `M:1359` |
| `layers.45.eh_proj.weight` | [4096, 8192] | BF16* | no module; dropped `M:1359` |
| `layers.45.input_layernorm.weight`, `.post_attention_layernorm.weight` | [4096] | BF16* | trunk equivalent `M:1274-1275`; dropped `M:1359` |
| `layers.45.self_attn.*` | a complete MLA + DSA indexer set as section 7/8 (FP8 q_a/q_b/kv_a/o_proj, BF16* kv_b, indexer) | as 7/8 | trunk equivalent `M:1092-1131`, `M:761-771`; dropped `M:1359` |
| `layers.45.mlp.*` | router + 288 experts + shared expert as sections 9/10 | as 9/10 | trunk equivalent `M:191-198`; dropped `M:1359` |
| `layers.45.shared_head.norm.weight` | [4096] | BF16* | no module; dropped `M:1359` |

Absent: any `hc_*` tensor (the block has no mHC), `embed_tokens`, `shared_head.head` (so it can only use the trunk's
`embed_tokens` and `lm_head`). **HF drops the whole block on load** (`_keys_to_ignore_on_load_unexpected =
[r"layers\.45\.", r"layers\.\d+\.shared_head\."]`, `M:1359`) and has no MTP forward. There is no HF golden for MTP
(open point O1).

**Forward (crow-nest #182, 2026-10-09):** the formula the serving stacks run (vLLM, SGLang, llama.cpp PR #29928 at
`b9acf138`) is `n = shared_head.norm(z)`, `z = y + MoE(post_attention_layernorm(y))`, `y = x + MLA_DSA(input_layernorm(x))`,
`x = eh_proj([enorm(embed(t_{i+1})) | hnorm(h_i)])` with `h_i` the trunk's post-final-norm row and the trunk's `lm_head`
on `n`; one stream, no mHC, the block's own full indexer and cache. Sources, the differences between them (row 0, the
paper's concat order) and the reference are [glm5-mtp.md](glm5-mtp.md); the reference is `oracle/glm5_mtp.py`, not HF.

## 12. HF vs llama.cpp (`649dcb103`) — differences

HF applies in every row. Rows marked *equivalent* are reformulations with identical math (different rounding); they matter for
tolerances and for the cache layout, not for the result.

| # | topic | HF | llama.cpp | kind |
|---|---|---|---|---|
| D1 | MLA cache | expanded K and V per head, 64 × (256+256) = 32,768 values/token/layer (`M:1175-1179`) | normalised latent `c`, 512 values/token/layer (`g5:18-19`, `g5:960`), `wk_b` absorbed into q (`g5:941-945`), `wv_b` after attention (`g5:984`, `g5:997`); `kv_b` split into `k_b` [64, 512, 256] / `v_b` [64, 256, 512] at conversion (`glm.py:540-550`) | equivalent; cache 64× smaller |
| D2 | indexer cache and pooling | 257 values/token `[k|gate|valid]` (`M:803`); pools recomputed every step, probabilities cast to BF16, BF16 weighted sum (`M:962-967`) | 384 values/token `[k|gate|pooled]` in `type_k` (default F16) (`hyb:57`, `g5:792-795`); a pool is computed once when complete, in F32 (`g5:800-821`) | equivalent math; precision differs |
| D3 | indexer score scaling | `relu(q·k·128^-0.5)` × `w·32^-0.5` (`M:826`, `M:829`) | `relu(q·k)` × `w·(128·32)^-0.5` (`g5:837-853`) | equivalent (scale > 0 commutes with relu) |
| D4 | router normalisation | `Σw + 1e-20` (`M:180`) | `clamp(Σw, 6.103515625e-5, ∞)` (`graph:2146-2150`) | differs only when Σ of 8 sigmoids < 6.1e-5 |
| D5 | mHC mixing precision | `post`/`comb` cast to BF16, stream mix in BF16 (`M:1316-1318`, `M:1325-1327`); collapse in f32 then BF16 (`M:294`) | everything F32 (`g5:391-548`) | precision |
| D6 | KDA prefill chunk | 64 (`M:488`) | 16 for KDA (`dn:61`) | equivalent (exact chunk algebra) |
| D7 | KDA conv state | last 4 pre-conv rows, model dtype (`CU:1066-1069`) | last 3 = `d_conv − 1` rows, F32, plus `n_rs_seq` rollback copies (`g5:203-226`, `lm:2476-2477`) | equivalent |
| D8 | DSA mask | top-k mask only (`M:1230-1246`) | top-k mask + causal `kq_mask` (`g5:905-907`) | equivalent (the indexer returns causal indices only) |
| D9 | MTP | block dropped, no forward (`M:1359`) | block loaded (`g5:177-184`), MTP graph throws "not implemented yet" (`g5:189-191`); MTP memory = one DSA layer with its own indexer (`lm:2458-2463`); the draft head would be fed the post-`output_norm` hidden state (`g5:666-670`). Since PR #29928 (`b9acf138`, 2026-10-07) the graph exists: [glm5-mtp.md](glm5-mtp.md) | HF has no reference; the formula of record is the serving stacks' (#182, O1) |
| D10 | stored precision (recipe, not math) | FP8 for q_a, q_b, kv_a_proj_with_mqa, o_proj (DSA), dense MLP, all experts (sections 7, 10; dequant `FP8:1003-1048`); the rest BF16*, four kept f32 at run time (`M:1358`) | converter forces `hc_*`, kpool gate/ape, `ssm_a`, `ssm_dt`, `exp_probs_b` to F32 (`glm.py:575-579`); `llama-quant.cpp` never quantises `hc_*`, indexer, `ssm_f_*`, `ssm_g_*`, `ssm_beta`, `attn_kv_a_mqa`, `attn_k_b`, `attn_v_b`, and keeps `attn_q_a`, `attn_q_b`, `nextn.eh_proj` at Q8_0 or higher | input for the step-4 keep set |
| D11 | parameter representation | `A_log`, `dt_bias` as stored (`M:315`, `M:314`) | `ssm_a = −exp(A_log)` (`glm.py:562-564`), `dt_bias` → `ssm_dt.bias` (`glm.py:566-567`), forget gate `sigmoid(−(g·ssm_a))·lb` (`g5:709-719`) | equivalent |

Checked equal (no row needed): mHC eps placement and Sinkhorn order (`g5:418-453` vs `M:286-290`), mHC input norm eps
1e-5 (`g5:472`), final unweighted mean then norm (`g5:663-666`), SwiGLU clamp on dense, shared and routed FFN
(`graph:1840-1844`, `graph:2234-2238`, ggml `swiglu_clamp`: `min(gate, l)`, `clamp(up, ±l)`; limit written for all blocks
`glm.py:514-516`), KDA l2norm `x/√(Σx²+1e-6)` (`models.h:14-17`) and q scale `128^-0.5` (`dn:45-47`), gated norm order,
indexer LayerNorm eps 1e-6 (`glm.py:468`), MLA scale `256^-0.5` (`g5:925`), pool alignment to the sequence's first position and
tail `(p − pos_min + 1) mod 4` (`hyb:788-805`, `hyb:1267`), kpool `ape` layout [4, 128] ↔ ggml {128, 4} (`g5:154`, `g5:807`),
selection width 512 pools + 3 tail = 2051 (`g5:328-329`).

## 13. Cache and state per token, per layer (derived)

| state | per token per layer | bytes | 11 DSA layers × 200,000 tokens | HF |
|---|---|---|---|---|
| MLA latent `c` (engine layout, llama.cpp) | `kv_lora_rank` = 512 values | 1,024 B at BF16/F16 | 1.126 B values = 2.25 GB | latent `M:1170-1172` |
| MLA HF layout (reference only) | `64 × (256 + 256)` = 32,768 values | 65,536 B at BF16 | 72.1 B values = 144 GB | `M:1175-1179` |
| indexer, HF layout | `128 + 128 + 1` = 257 values | 514 B at BF16 | 0.565 B values = 1.13 GB | `M:800-810` |
| indexer, llama.cpp layout | `3 × 128` = 384 values | 768 B at F16 | 0.845 B values = 1.69 GB | same inputs `M:797-800` |
| indexer, minimal (derived) | pooled key 128 per 4 tokens = 32 values, + ≤ 3 pending `[k|gate]` rows per layer per sequence | 64 B at BF16 | 0.070 B values = 0.14 GB | `M:961-967`, `M:824-826` |

The minimal indexer layout is exact only if the pooled key is computed with HF's casts (`M:964-967`); raw indexer keys and gates
of a completed pool are never read again (`M:824`, attention reads the MLA cache). The MTP block adds one more DSA layer of each
when it runs.

Per sequence, independent of the token count:

| state | size | bytes |
|---|---|---|
| KDA recurrent state | 64 × 128 × 128 = 1,048,576 f32 per layer (`M:454-461`, `M:726`) | 4 MiB per layer, 136 MiB for 34 layers |
| KDA conv window | 3 × 8192 × (4 − 1) = 73,728 values (llama.cpp F32, `g5:203-204`); HF keeps 4 rows = 98,304 values in BF16 (`CU:1066-1069`) | 288 KiB per layer F32, 9.56 MiB for 34 layers |

Activations between layers are 4 streams × 4096 = 16,384 values per token (`M:1477`), four times a single-stream model.

## 14. Config keys HF never reads

`grep -c` over `M` and `C` = 0 for each key.

| key (cfg value) | llama.cpp | note |
|---|---|---|
| `indexer_rope_interleave` (true) | unused | no RoPE dims exist (`qk_rope_head_dim` 0, `index_head_dim` 128 all NoPE; indexer q/k at `M:797-798`); see O4 |
| `index_kpool_compress` (true) | unused | both always compress (`M:961-967`) |
| `index_share_for_mtp_iteration` (true) | unused | MTP only, HF drops MTP (`M:1359`); see O1 |
| `first_k_dense_replace` (3) | `glm.py:509` (leading dense blocks) | HF uses `mlp_layer_types` instead, same 3 layers (`M:1270-1272`, `C:160-163`) |
| `num_nextn_predict_layers` (1) | `glm.py:420` | HF has no MTP (`M:1359`) |
| `moe_router_dtype` (float32) | unused | HF computes the router in f32 regardless (`M:160`) |
| `topk_method` (noaux_tc) | unused | HF's router is noaux_tc by construction (`M:162`, `M:178`) |
| `mla_use_nope` (true) | asserted `glm.py:482` | HF validates `qk_rope_head_dim == 0` instead (`C:225-228`) |
| `scoring_func` (sigmoid) | unused (gating defaults to sigmoid, `g5:30-33`) | HF hard-codes sigmoid (`M:161`) |

## 15. Open points

- **O1 MTP forward.** No reference: HF drops layer 45 (`M:1359`), llama.cpp throws (`g5:189-191`). Unknown: concat order and
  inputs of `eh_proj` (DeepSeek-V3 convention would be `eh_proj([enorm(embed(t+1)) | hnorm(h_t)])`), which `h_t` (llama.cpp
  feeds post-`norm`, `g5:666-670`), how a block without `hc_*` tensors joins a 4-stream trunk, what
  `index_share_for_mtp_iteration` means. Step 21 needs a source of record before any MTP kernel.
  **Settled 2026-10-09 (crow-nest #182)** from vLLM, SGLang and llama.cpp PR #29928 (`b9acf138`): `eh_proj([enorm(e) |
  hnorm(h)])`, embedding first (the DeepSeek-V3 paper writes h first; every code base puts e
  first, and h first gives 0 / 89 draft agreement on the real weights), `h` = the trunk's post-`norm` row, a plain pre-norm residual layer on one stream (no mHC),
  `shared_head.norm` then the trunk's `lm_head`; `index_share_for_mtp_iteration` = draft steps 1+ reuse step 0's
  top-k of the block's own indexer. Still open: no vendor golden (the reference is `oracle/glm5_mtp.py` over HF's
  blocks); the stacks differ at row 0 (vLLM zeroes its embedding, llama.cpp prepends a `(t_0, 0)` row). Details
  [glm5-mtp.md](glm5-mtp.md).
- **O2 stored dtypes.** idx has no dtypes; the non-FP8 tensors are BF16 per cfg `dtype`, but A_log, dt_bias,
  e_score_correction_bias, `hc_*`, conv weights, norms and the `weight_scale_inv` tensors may be F32 on disk. The byte balance
  leaves 1,127,845,052 B for the vision tower plus anything wider than BF16. The converter (steps 4/5) reads the safetensors
  headers and settles it.
- **O3 conv weight rank.** HF concatenates `{q,k,v}_conv1d.weight` along dim 0 into `[24576, 1, 4]` (`CM:571-579`, `M:608-615`),
  which implies `[8192, 1, 4]` per tensor; llama.cpp accepts rank 2 or 3 (`glm.py:553-560`). Settled by the same header read.
- **O4 indexer RoPE.** cfg says `indexer_rope_interleave: true`, but neither HF (`M:797-798`) nor llama.cpp (`g5:779-789`) rotates
  anything in the indexer, and there are no rope dims. Both agree, HF stands; only a vendor reference could show otherwise.
- **O5 reference precision.** HF's BF16 casts (mHC mix `M:1316`, pooled keys `M:964-967`, KDA output `M:478`) are part of the
  reference. An engine path in F32 will not match bit for bit; the tolerance belongs to the PREREG gates, not to this page.
- **O6 padded batches.** HF aligns pools to the first valid token per row (`M:940-947`); single unpadded sequences align at
  position 0. Crow decodes one sequence, so only a batched golden in step 7 would exercise this.
