# Numerics diff: GDN normalization + QSA sparse attention vs HF reference and llama.cpp

Issue #89. Produced 2026-09-20 by line-by-line reading of three implementations.
No engine code was changed; probes listed in §3 were added as new `engine/src/bin/*_probe.rs` files.

**Sources compared (all read locally):**

- **crow-nest** — `engine/src/kernels.rs` (CUDA `KERNEL_SRC`), `engine/src/gen.rs`, `engine/src/manager.rs`, `engine/src/geo.rs`.
- **HF reference** — transformers 5.16.1 in `.venv-oracle/lib/python3.14/site-packages/transformers/models/qwen4_exp/modeling_qwen4_exp.py` (the oracle venv; `geo.rs` names qwen4_exp the text tower; `config.json` `model_type = qwen4_exp`). The GDN + indexer are **pure-torch inside the modeling file** (no `fla` import); the `l2norm` helper at modeling line 259 is documented as "intended to align with the l2norm implementation in the FLA library".
- **llama.cpp** — shallow clone at `/tmp/llamacpp` (master, 2026-09): `src/models/qwen4exp.cpp` (this arch **is** implemented there, including the QSA indexer), `src/models/qwen3next.cpp`, `src/models/models.h`, `src/models/delta-net-base.cpp`, `src/llama-graph.cpp`, `src/llama-memory-hybrid-idx.cpp`.
- **config** — `models/Qwen3.8-Flash-Next-original/config.json` (text_config).

**Verdict summary: 17 MATCH / 0 MISMATCH / 3 UNVERIFIABLE** (tie-break order vs `torch.topk`; GGUF-side +1 weight fold; RoPE table bit-provenance). Details and severities below. Every pinned constant checked against `config.json` agrees (see §4).

---

## 1. Diff table

Verdicts: **MATCH** = formula and constants identical (fp reduction order may differ — noted); **MISMATCH** = different formula/constant; **UNVERIFIABLE** = cannot be decided from the sources read, with reason.

| # | check item | crow-nest | HF reference | llama.cpp | VERDICT |
|---|---|---|---|---|---|
| 1 | Full-attn q/k norm form + eps | per-head RMSNorm over 256, `rsqrtf(Σx²/256 + 1e-6)`, weight `(1+w)`, f32 — `rmsnorm_1pw` kernels.rs:1431-1446 | `Qwen4ExpTextRMSNorm`: `x.f32()·rsqrt(mean(x²)+eps)·(1+w)`, eps=`rms_norm_eps`=1e-6 — modeling:158-181, applied modeling:810-811 | `build_norm(..., LLM_NORM_RMS)` + direct weight mul — llama-graph.cpp:1591,1604-1605; qwen4exp.cpp:810-817 | **MATCH** (llama.cpp folds `+1` at GGUF conversion — see U2) |
| 2 | GDN q/k l2norm formula (post-#28068 form) | `x·rsqrtf(Σx² + 1e-6)`, q additionally `·rsqrtf(128)` — `l2norm_repeat` kernels.rs:1589-1602 | `l2norm`: `x·rsqrt(Σx² + 1e-6)`; then `query = query·(1/√128)` — modeling:259-262, 279-281, 295-296 / 361-362, 369-370 | `ggml_scale(ggml_rms_norm(x, eps/n), 1/√n)` ≡ `x·rsqrt(Σx²+eps)` algebraically; q-scale inside the GDN op — models.h:14-18 ("ref: …pull/28068"), qwen4exp.cpp:956-968 | **MATCH** — the #28068 class of bug (max instead of rsqrt) is **not** present in crow-nest |
| 3 | GDN out-norm (gated) | `w[d]·x·rsqrtf(Σx²/128 + 1e-6)·sigmoid(z)`, f32, weight applied directly — `rmsnorm_gated` kernels.rs:1756-1772 (fused twin 3405-3419) | `RMSNormGated`: `weight·(x·rsqrt(mean+eps))·ACT(gate)`, weight direct (ones-init), activation=`config.output_gate_type`=**sigmoid** — modeling:185-201, 437-439; config.json | `rms_norm` then `ggml_sigmoid(gate)` mul — qwen4exp.cpp:476-486 ("sigmoid output gate, not silu") | **MATCH** |
| 4 | GDN recurrence (decode/step) | `S·=g; kv=Σ_dk S·k; δ=(v−kv)β; S+=k⊗δ; o=Σ_dk S·q`, f32, intrinsics pin op order — `delta_rule_persist` kernels.rs:1619-1644 / `_r` 1687-1727 / `delta_rule_step_r` 1728-1755 | identical step in f32 — modeling:381-392 (`torch_recurrent_gated_delta_rule`) | same recurrence via fused GDN ops — delta-net-base.cpp:527-572, qwen4exp.cpp:968 | **MATCH** (per-element op order equal; 128-sum reduction order differs — f32 reassociation only) |
| 5 | GDN prefill algorithm | token-recurrent over the whole chunk (`delta_rule_persist`), same op order as decode | **chunked** UT-transform, chunk_size=64 — modeling:266-344 (`torch_chunk_gated_delta_rule`, chosen at modeling:539-550) | chunked fused (`LLM_FUSED_OP_GDN_CH`) — delta-net-base.cpp:568-572 | **MATCH** as formulas / **numerics-class difference**: recurrent vs chunked reassociation (see P2) |
| 6 | GDN β and decay g | `β=sigmoid(b)`, `g=−expf(A_log)·logf(1+expf(a+dt_bias))`, f32 — `beta_g` kernels.rs:1603-1617 | `β=b.sigmoid()`, `g=−A_log.float().exp()·softplus(a.float()+dt_bias)` — modeling:517-519 | `sigmoid`, `softplus`, `mul` — qwen3next.cpp:443-455 | **MATCH** (notes: torch `softplus` linearizes above 20 (≤2e-9 rel); HF computes β in bf16, crow in f32) |
| 7 | GDN GQA mapping (k→v heads) | `khead = vhead / 3` (16 k-heads → 48 v-heads) — kernels.rs:1594; norm computed once per k-head, value replicated | `repeat_interleave(3, dim=2)` — modeling:520-522 | `ggml_repeat_4d` interleave — qwen3next.cpp:514-530, qwen4exp.cpp:960-964 | **MATCH** |
| 8 | Attention scale value + application point | `0.0625` (=256^−0.5) at kernels.rs:1860, 1932, 2096, 2194 (`g_score`), 3709 (+`attn_merge` divide); applied to the raw dot `p[j]=acc·scale` **before** max-subtraction (1871/1950/2117/3726) | `scaling = head_dim**-0.5` = 0.0625 — modeling:766; eager: `matmul(q,kᵀ)·scaling` then softmax (max-subtract inside) — modeling:745-749 | `kq_scale = 1/√n_embd_head` (or `f_attention_scale` metadata) passed to `ggml_soft_max_ext` — qwen4exp.cpp:841-842, llama-graph.cpp:2694 | **MATCH** (positive scale commutes with max-subtraction; all three apply it pre-softmax) |
| 9 | Softmax accumulation precision | f32 end-to-end: warp-shuffle dot, shared-mem max/sum trees, `expf`, per-element IEEE `p[j]/sum` — kernels.rs:1869-1901; router f32 — 2945-2953 | eager softmax `dtype=torch.float32` — modeling:749; router `softmax(dtype=float)` — modeling:910 | kq accumulation forced f32 (`ggml_prec_set_acc(kq, GGML_PREC_F32)`) — llama-graph.cpp:2665; soft_max in f32 | **MATCH** on precision class; reduction order differs per engine (expected, not a formula item) |
| 10 | RoPE pairing + theta + partial split | pairs `(d, d+32)` for d<32 = rotate_half on the first **64 of 256** dims, dims ≥64 pass through; table `cos/sin[t·32+j]`, `inv=1e7^(−2j/64)` host f32 — `rope` kernels.rs:1784-1799, `rope_p` 1802-1819; manager.rs:300-311; `ROPE_PAIRS=32` geo.rs:26 | `partial_rotary_factor=0.25` → rotary dim 64; `inv_freq = 1/(θ^(arange(0,64,2)/64))`, θ=`rope_theta`=1e7; `emb=cat(freqs,freqs)`; `q_rope·cos + rotate_half(q_rope)·sin`, pass-through rest — modeling:108-116, 566-608; config.json | `ggml_rope_ext`/`ggml_rope_multi` with `n_rot`, `freq_base` from metadata — qwen4exp.cpp:716-723, 822-833 | **MATCH** (table bits vs torch: ULP-level, U3) |
| 11 | mrope (text path) | single T-plane table (all positions equal for text) | `apply_interleaved_mrope` mixes H/W planes at interleaved sections — modeling:121-155; **no-op when the 3 position planes are equal** (text-only decoding) | `ggml_rope_multi` with sections; text uses equal planes — qwen4exp.cpp:716-723 | **MATCH** for text; multimodal not in scope of this engine |
| 12 | GQA head mapping (full attention) | `kvh = head / 12` (24 q → 2 kv, contiguous blocks of 12) — kernels.rs:1850, 1918, 3695; `store_kv` layout kernels.rs:1832-1835 | `repeat_kv` expand+reshape ≡ head h ← kv `h // 12` — modeling:720-729, 742-743 | standard llama.cpp GQA mapping in `build_attn_mha` | **MATCH** |
| 13 | QSA indexer q/k norm + eps | per-head RMSNorm over 128, `(1+w)`, eps 1e-6 — `rms128` kernels.rs:2336-2354 (q: gen.rs:2582; pooled k: gen.rs:2600) | `Qwen4ExpTextRMSNorm(128, eps=rms_norm_eps)` for q and pooled k — modeling:628-629, 651, 682 | `build_norm(..., LLM_NORM_RMS)` — qwen4exp.cpp:634-635, 643-645 | **MATCH** |
| 14 | QSA indexer geometry (budget/compress/block math) | budget 2048, ratio 4, block top-k 512 (`min(K, ncb)` blocks), `ncb=(pos+1)>>2` complete blocks, tail `(pos+1) mod 4` appended unconditionally, list = 4 tokens/block ascending + tail; `QSA_BLOCK_TOPK=512` geo.rs:89, `qsa_scores` kernels.rs:2421, `qsa_select` 2451-2586, dense shortcut 2596-2604, decode `ncb1` gen.rs:3941-3946 | `token_budget=2048`, `compress_ratio=4`, `block_topk=2048//4=512`; `num_complete_blocks = visible//4`; `topk(min(block_topk, ncb))` blocks → 4 tokens each; `tail = visible[4·ncb:]` concatenated — modeling:620-622, 672-701; config.json | token-level top-k with `width = min(n_kv, top_k + r − 1)` over per-token expanded block scores ("the reference returns indexer_top_k + compress_ratio - 1: whole blocks plus the tail") — qwen4exp.cpp:678-682; incomplete groups not pooled — llama-memory-hybrid-idx.cpp:354 | **MATCH** (all three select whole blocks + unconditional tail; set-equal) |
| 15 | QSA indexer score formula | `(Σ_h relu(q_h·k̂_b))·rsqrtf(128)` in f32 — `qsa_scores` kernels.rs:2415-2442 / `qsa_scores_par` 3644-3674 | `relu(q·kᵀ)`, sum over the 4 heads, `/√128`, f32 matmul — modeling:690-693 | relu per head then head-sum, **no `/√128`** — qwen4exp.cpp:649-661 | **MATCH** (scores feed only top-k, a positive scale is rank-invariant; llama.cpp omitting it is harmless — noted for completeness) |
| 16 | QSA pooled-key pipeline order | pool **raw** keys → RMSNorm → RoPE at block start `p=4·b` — `pool4_cache` kernels.rs:2379-2390 (raw ring), gen.rs:2592-2607; `pos_mul4=4`, `pos_base_b4`=block index (rope64 kernels.rs:2368 `p=(pos_base+t)·4`) | raw keys cached (`update_indexer`), `mean(dim=1)` in f32 → **cast back to bf16** → `k_layernorm` → RoPE at `group_starts` (block start) — modeling:654-688 | "cached indexer keys are raw: pooling precedes norm and rotation"; mean → rms_norm → rope at block-start pos — qwen4exp.cpp:593-641, llama-memory-hybrid-idx.cpp:474-493 | **MATCH** (precision note: HF rounds the pooled mean to bf16 before the norm; crow keeps f32 — crow strictly finer) |
| 17 | QSA pooling formula | `(k0+k1+k2+k3)·0.25` over 4-aligned raw rows — kernels.rs:2388-2389 | `key_groups.float().mean(dim=1)` — modeling:681 | slice-add ×4 then `ggml_scale(1/r)` — qwen4exp.cpp:620-631 | **MATCH** |
| 18 | QSA raw-key ring + wraparound | ring = `round4(chunk+4)` rows (default; `CROW_QSA_FULL=1` → context), `ring%4==0` ⇒ a block's 4 rows never straddle the wrap; row = `pos % ring`; pooled immediately at block completion so the pooled cache is full-length — manager.rs:100-113, `qk_k_append` kernels.rs:2392-2401, `pool4_cache` 2386 | raw keys kept for **all** positions (cache) — modeling:654-655 | full indexer cache — qwen4exp.cpp:593-596 | **MATCH** (mechanism is engine-eigen but numerics-neutral: pooled values are a pure function of the same 4 raw rows) |
| 19 | QSA selection tie-break | exact radix/`ordkey` top-k, ties resolved **lowest block index**, ascending emit — `qsa_select` kernels.rs:2447-2542, `qsa_select_fast` 2589+, `qsa_select_par_h/_e` 2741-2912 ("Output rule reproduced byte for byte from qsa_select_fast") | `torch.topk` — modeling:695 (**tie index order unspecified on CUDA**) | `ggml_top_k` over expanded token scores — qwen4exp.cpp:679 (tie order = argsort implementation detail) | **UNVERIFIABLE** vs torch (U1) — engine-side determinism proven by probe P1 |
| 20 | QSA list semantics vs HF mask | gathered list `[4·b ascending…] + tail`, count `sel_n`; out-of-range clamps `tok<0→0, tok≥tmax→tmax−1` are defensive only (selectors never emit them) — kernels.rs:1865-1866, 2571, 2581 | selected indices scattered into a boolean/float mask over kv; `-1` padding scattered to `kv_length` and dropped — modeling:704-717 | top-k indices unmask rows of an otherwise `−inf` mask — qwen4exp.cpp:719-752 | **MATCH** (set-equal; padding never attended in any engine) |
| 21 | Prefill-vs-decode row equality (QSA attention) | prefill `attn_sel*`: normalize each weight `p[j]/sum` then `o += w·v` in list order; decode `attn_sel_split`+`attn_merge`: unnormalized partials `(m,l,o)`, merged `o/L` once — kernels.rs:1886-1901 vs 3740-3791 | single eager/sdpa path (mask-based), no engine-internal split | single graph path, no engine-internal split | **MATCH** at formula level (same token set, f32); bit-level differs by design — measured by probe P4 |
| 22 | Router softmax + top-10 + renorm | f32 softmax with max-subtract; iterative argmax ×10 with `atomicMin` ⇒ ties **lowest expert index**; weights renormalized `s_ws/s_sum` over the 10 — `router_top10` kernels.rs:2915-2990 | `softmax(dtype=float)` → `torch.topk(10)` → `/= sum` when `norm_topk_prob` (true) — modeling:907-916 | `build_moe_ffn` with `expert_weights_scale` + norm top-k — qwen4exp.cpp:1015-1030 | **MATCH** (renorm exact; tie-break shares U1's caveat — exact f32 prob ties are rare on continuous logits) |
| 23 | Attention output gate | `core / (1+expf(−g))` = `core·sigmoid(gate)` — `gate_mul` kernels.rs:2308-2313 (+fused `gate_mul_q` 3421+); q/gate split: q first 256, gate second 256 per head — `split_qg` kernels.rs:1775-1783 | `chunk(q_proj.view(…, head_dim·2), 2)` = q then gate; `attn_output·sigmoid(gate)` — modeling:805-808, 836 | `ggml_sigmoid(gate)` mul; gate = second half of wq — qwen4exp.cpp:791-799, 851-853 | **MATCH** |
| 24 | Hyper-connection (HC) mixing | group RMSNorm per 2560-stream `(1+w)` — `rms_group` kernels.rs:1411-1429; mix = `sigmoid(up(silu(down(x)·0.25)))` — `silu_div4` 1448-1454 + `sigmoid_el` 1455-1459; mean over 4 streams `·0.25` — `mix_streams` 1466-1474; inject = `2·sigmoid(w·0.25)` — `sig2_div4` 1460-1465 + `inject_residual` 1475-1483 | `hc_norm = RMSNorm(4·2560, group_size=2560)`; `silu(down(x)/hc_count)` → `sigmoid(up(·))` → `.mean(dim=-2)`; `injection = 2·sigmoid(block_inject_weight(x)/hc_count)`; residual `hyper_input + out·injection` per stream — modeling:941-969, 1236-1243 | `silu`/`sigmoid` lora chain with `/hc` scales — qwen4exp.cpp:285-345 | **MATCH** |

### Verdict tally

**17 MATCH, 0 MISMATCH, 3 UNVERIFIABLE** (U1 tie-break order, U2 GGUF +1 fold, U3 RoPE table bits — all minor; see §2).

Two systematic, engine-wide observations that are **not** mismatches but bound every comparison:

- **dtype path**: HF computes elementwise math in bf16 (model dtype) and upcasts only norms/softmax/delta-rule to f32; crow-nest dequantizes once and stays f32 throughout; llama.cpp runs its native f16/f32 mix. crow-nest is everywhere ≥ HF precision at the sites compared, so no crow-side correction is indicated by this diff.
- **reduction order**: every f32 sum (softmax Z, 128-dim dots, 10-rank MoE combine) has a different but deterministic order per engine. This is bit-level noise (~1 ulp/class), the existing parity gates' domain — not a formula divergence.

---

## 2. UNVERIFIABLE items, with reasons and severity

**U1 — top-k tie-break vs `torch.topk` (severity: MEDIUM for long contexts, ZERO for current oracle rows).**
crow-nest resolves exact-f32 score ties by lowest index in all three selector kernels (kernels.rs:2513-2542, 2747-2749) and the router (kernels.rs:2965). `torch.topk`'s index order among equal values is an implementation detail of the CUDA sort, not a contract (modeling:695, 911). Exact ties are measure-zero on continuous router logits, but QSA **block scores can tie exactly** (e.g. relu-clamped zero scores, identical pooled keys after e4m3/bf16 rounding), and in the sparse regime (ncb > 512, i.e. prompts > 2048 tokens) a tie straddling the budget boundary selects a *different block* → a different 4-token set in the attention mask. The oracle rows (298/607 tokens) are entirely in the dense regime (ncb ≤ 151 < 512), so no oracle row can detect this — that is precisely the #68 long-context coverage gap. Probe P1 pins the engine side; the torch side needs one venv experiment (P1b).

**U2 — llama.cpp GGUF-side `+1` fold for zero-centered RMSNorm (severity: INFO).**
llama.cpp's graph multiplies RMSNorm weight directly (llama-graph.cpp:1604-1605), so the `(1+w)` fold must happen at GGUF conversion for the zero-centered qwen4exp norms (HF applies `1+w` at runtime, modeling:177). The converter script is not present in the shallow clone (`convert_hf_to_gguf.py` is a 312-line shim; the real converter moved). The fold is the long-standing convention for this family (qwen3next historically did `weight + 1.0` at conversion), so I file this as an unverified micro-detail of llama.cpp, not of crow-nest. No action.

**U3 — RoPE table bit-provenance (severity: NEGLIGIBLE).**
crow-nest builds the table in host f32: `10_000_000f32.powf(−2j/64)` then `t·inv`, `(cos,sin)` f32 (manager.rs:300-311). HF computes `base ** (arange/64)` in torch f32 then f32 matmul with positions (modeling:115, 125-136). Same formula, same f32 class, but `powf` vs torch's pow can differ by ≤1-2 ulp per `inv_freq`, which after `t·inv` and cos/sin is still ~ulp-level in the rotated 64 dims. Formula MATCH; exact-bit equality unverified and almost certainly immaterial. Optional probe P3.

---

## 3. Probe plan (for every UNVERIFIABLE, plus the numerics-class items)

Pattern: the existing `engine/src/bin/router_probe.rs` / `qsa_probe.rs` shape — no model, no container, no engine lock; `cuda::compile(kernels::KERNEL_SRC)` + `launch_v` on synthetic buffers; deterministic `sample::Rng`; explicit pass line. All new files; no existing source touched.

### P1 — `qsa_tie_probe` (implements U1's engine side) — **IMPLEMENTED AND RUN (2026-09-20, RTX 5090): 7/7 PASS**
- **Input**: one score row per case, `ncb ∈ {400..3000}`, K = 512 (production `QSA_BLOCK_TOPK`), `cap = 65536`, `sel_max = QSA_SEL_MAX` (2051); constructed so a group of 8-24 blocks shares one **bit-identical** f32 value that straddles rank 512 (i.e. the K-th largest value is tied across the boundary), plus a boundary-exact case (tie group exactly consumed by the fill), an all-tied case, and dense-regime cases; tails 0..3.
- **Reference (host Rust)**: select blocks by (value desc, index asc) — the documented-intent torch rule — then emit 4 tokens per selected block in ascending block order + ascending tail (the engines' emit contract).
- **Run**: `qsa_select`, `qsa_select_fast`, and the `qsa_select_par_h`+`qsa_select_par_e` pair (via `gen::launch_qsa_par_e`, h1 zeroed between cases), grid/block exactly as gen.rs launches them.
- **Result**: all three selectors byte-identical to the reference on every case, including all-tied and boundary-exact; ties resolved lowest-index; h1 re-zeroed; poison beyond `sel_n` untouched. The engine side of U1 is pinned deterministic; only the torch side (P1b) remains open.
- **Also found** (see §6): the plain radix `qsa_select` has no dense shortcut and breaks for K > ncb — latent, env-gated.
- **Command**: `LD_LIBRARY_PATH=$HOME/.local/share/crow/cuda/lib cargo run --bin qsa_tie_probe` (from `engine/`).

### P1b — torch tie-order experiment (venv, no code) — PLANNED
`.venv-oracle` python: build the same planted-tie rows, run `torch.topk(row, 512)` on CUDA (and CPU for reference), record the index order among the tied group straddling 512, compare to lowest-index. If torch's order differs on exact ties, quantify the QSA consequence: fraction of tied-boundary rows in long-context generations (needs > 2048-token rows), worst-case softmax-mass shift of swapping one 4-token block. Feeds #68's suspect list.

### P2 — GDN prefill numerics-class probe (recurrent vs HF chunked) — PLANNED
- **Input**: normed q/k (post-`l2norm_repeat`, q pre-scaled 1/√128), v, g, β for T ∈ {64, 512, 4096}, 48 heads, drawn O(1) uniform; also one real layer-3 oracle dump if available.
- **Reference**: `torch_chunk_gated_delta_rule` from the oracle venv (chunk 64) on the same f32 tensors, plus the step-recurrent torch form as a second reference (they also differ from each other — that difference IS the numerics-class bound).
- **Run**: crow `delta_rule_persist_r` on the same inputs (the probe pattern compiles `KERNEL_SRC` directly).
- **Measure**: per-head max_abs and rel_L2 of `core_attn_out`, and of the final recurrent state S (state error compounds across chunks — the long-context risk).
- **Tolerance**: rel_L2 ≤ 1e-5 expected for outputs; report-only for S with growth-vs-T noted. If ≥ 1e-4, the GDN prefill class is a real residual contributor and chunking crow's prefill (or accepting the class) becomes a decision with numbers.
- **Command**: `cargo run --release --bin gdn_chunk_probe` (to be created in a follow-up; needs a small torch-side dumper first).

### P3 — RoPE table ULP probe — PLANNED (cheap, optional)
Host-only Rust: recompute the manager.rs table and an f64-rounded reference over the full 262144 positions × 32 pairs; report max ulp distance of `inv_freq`, `cos`, `sin`; plus one golden-vector check of the `rope` pairing formula vs a host rotate_half. Pass: ≤ 2 ulp. (`cargo run --bin rope_table_probe`.)

### P4 — `attn_path_probe` (prefill-vs-decode QSA attention row equality) — **IMPLEMENTED AND RUN (2026-09-20, RTX 5090): 3/3 PASS**
- **Input**: one query row (24 heads × 256, O(1)), a bf16-mode KV cache (`tmax = 4096`) filled with distinct bf16-truncated f32 values, and one selected list of n = 2051 ascending distinct tokens (< tmax) — the sparse-regime worst case.
- **Run**: prefill form `attn_sel` vs decode form `attn_sel_split` (S = 1, 2, 8 splits) + `attn_merge`, launched exactly as gen.rs does (grid (24, T[, S]), block 256, device scalars).
- **Measured floor**: rel_L2 1.08e-6 (S=1), 8.98e-7 (S=2), 7.97e-7 (S=8); max_abs ≈ 1.1e-7 on O(1) outputs. The prefill-vs-decode divergence is real but two orders below the 1e-5 line and three below the layer-3 residual (2.9%) — it cannot explain the residual.
- **Command**: `LD_LIBRARY_PATH=$HOME/.local/share/crow/cuda/lib cargo run --bin attn_path_probe` (from `engine/`).

### Existing gates that already cover other rows
- `parity` / selftest goldens (layer-0/layer-3 max_abs/rel_L2) — the promotion path for any future fix; row 5/21's engine-internal bit-differences are already inside those gates' tolerance.
- `qsa_probe` — par-vs-fast selector byte equality on random + narrow + tied distributions (complements P1's reference-defined semantics).
- `router_probe` — gemv vs dense router forms, top-10 set agreement.

---

## 4. Constant cross-check (config.json vs crow-nest pins)

| constant | config.json | crow-nest | site |
|---|---|---|---|
| `rms_norm_eps` | 1e-06 | `1e-6f` (every norm) | kernels.rs:1426, 1444, 1599, 1769, 2352, 3413 |
| `rope_theta` | 10 000 000 | `10_000_000f32` | manager.rs:303 |
| `partial_rotary_factor` | 0.25 → 64 of 256 | `ROPE_PAIRS = 32` | geo.rs:26 |
| `head_dim` | 256 | `AHD = 256`; scale 0.0625 | geo.rs:18, kernels.rs:1860 |
| `num_attention_heads` | 24 | `NQ = 24` | geo.rs:16 |
| `num_key_value_heads` | 2 | `NKV = 2`; `kvh = head/12` | geo.rs:17, kernels.rs:1850 |
| `output_gate_type` | sigmoid | `1/(1+e^{-z})` | kernels.rs:1770, 2312 |
| `hidden_act` | silu | `x/(1+e^{-x})` | kernels.rs:1448-1449, 1548 |
| `indexer_budget` | 2048 | `QSA_BLOCK_TOPK·QSA_COMPRESS` | geo.rs:88-90 |
| `indexer_compress_ratio` | 4 | `QSA_COMPRESS = 4` | geo.rs:88 |
| `indexer_n_heads` / `kv_heads` / `head_dim` | 4 / 1 / 128 | `QSA_HEADS/KVHEADS/HD` | geo.rs:84-86 |
| `linear_{k,v}_heads`, head dims | 16 / 48 / 128 / 128 | `GDN_KHEADS/VHEADS/GD` | geo.rs:13-15 |
| `num_experts` / per-tok | 512 / 10 | `E` / `TOPK` | geo.rs:7-8 |
| `hc_count` / `hc_lowrank` | 4 / 320 | `HCN` / `LOWRANK` | geo.rs:4, 6 |
| `max_position_embeddings` | 262144 | `context` default; score cap 65536 blocks | geo.rs:163, gen.rs:2030 |

All pins agree with the config. The pinning *practice* (vs metadata-read) remains the T12 concern — the values themselves are correct today.

## 5. Findings beyond the diff (engine-internal, discovered by the probes)

**F1 — `qsa_select` (plain radix) breaks in the dense regime, K > ncb (severity: LOW — env-gated, unreachable by default).**
Found by `qsa_tie_probe` case "dense ncb400" on 2026-09-20: `sel_n = 3` instead of 1603. Mechanism: the radix threshold search (kernels.rs:2481-2494) computes `need = K - above` and scans buckets for `cum + c >= need`; with K = 512 > ncb = 400 the condition is never met, `B` stays at its 255 initializer, and the bitmap/fill logic degenerates. `qsa_select_fast` (kernels.rs:2596-2604) and `qsa_select_par_e` (2836-2842) both guard `if (K >= ncb)` with the dense shortcut and are correct. Production paths never reach the plain radix: prefill defaults to `qsa_select_fast` (`CROW_QSA_FAST` on) and decode to the par pair (`CROW_QSA_PAR` on). Reachable only via `CROW_QSA_FAST=0` **and** `CROW_QSA_PAR=0` **and** a prompt of ≤ 2048 tokens. Not a numerics mismatch vs HF (the default arms match HF exactly); a fix is out of scope for #89 (no engine code changed) — either the same two-line dense shortcut in `qsa_select` or dropping the radix variant once the par pair is proven final.

## 6. Conclusion for the acceptance metric

The #89 hypothesis — a wrong-formula constant or op in GDN normalization or QSA attention ("the engine numerics suspect of expert-requant.md §8") — is **acquitted on every line compared**: no MISMATCH was found, including the exact bug class of llama.cpp #28068 (crow-nest's `l2norm_repeat` was already the rsqrt form). Measured floors so far: the prefill-vs-decode QSA attention divergence is ≤ 1.1e-6 rel_L2 at the full sparse list length (P4), two orders below the 1e-5 line and three below the layer-3 residual; the engine-side QSA tie-break is deterministic lowest-index across all three selectors (P1). The remaining open risk is (a) the tie-break order vs `torch.topk` at exact score ties in the sparse regime (U1/P1b, untestable by the current dense-regime oracle rows), (b) the recurrent-vs-chunked GDN prefill numerics class (P2, planned), and (c) generic f32-order noise. If the layer-3 attention residual (2.9% rel_L2) does not move after P2 measures its floor, the residual is attributable to weight quantization (FP4/BF16 keeps), not attention/GDN formula drift — that attribution is exactly what these probes provide. Fixes, if ever needed, go through the standard env-gated → bit-parity promotion path and become model-read via #T12. One engine-internal latent defect was found and filed without fixing it (F1, §6).
