# layerdiff: per-layer activations of crow-nest vs llama.cpp at one position

The tooling that found the PLE row-offset bug of `#91` (fixed in `85a48e7`, 2026-09-23). It
compares the residual stream, the sub-block outputs and the final logits of crow-nest (CNQ4.5-M)
and llama.cpp (UD-Q2_K_XL) layer by layer at a corrupt site, so the first layer where the two
engines split is visible instead of guessed.

## Files

| file | what it does |
|---|---|
| `ldump.cpp` | llama.cpp example program. Decodes `ids[0..N-K]` without a callback, then the last `K` ids one at a time with a `cb_eval` that saves whitelisted per-layer tensors (`l_last-`, `hc_*`, `attn_output-`, `linear_attn_out-`, `ffn_*`, `ple_*`, `result_norm`, ...) plus the logits row to `LD_DIR/p<pos>/`. Env: `LD_IDS` (whitespace-separated ids), `LD_DIR`, `LD_K` (default 1); every other argument is a normal llama.cpp common argument (`-m`, `-c`, `-ctk`, ...). Build it inside a llama.cpp tree as an extra example target linked against `common` and `llama`. |
| `compare.py` | Reads `$LAYERDIFF_DIR/sites.json` and the dump directories `llama-<site>/p<pos>` and `cn-<arm>-<site>/p<pos>`, writes `compare-<arm>.json`: per layer cos / relative L2 of `h_out`, `attn_mixed`, `sub_out`, `x1`, `ffn_mixed`, `moe_out`, `routed`, `shared`, the layer-1 PLE contribution, per-stream cos, experts in common, and at the site the logprob margin correct - corrupt of both engines. Usage: `LAYERDIFF_DIR=... python3 compare.py <arm>`. |
| `cnq_ple.py` | Decodes layer-1 PLE n-gram rows straight from the CNQ container and from the GGUF (`per_layer_token_embd.weight`) and compares them with the embedding crow-nest gathered. Its `row()` deliberately reproduces the PRE-`85a48e7` read (byte `row * 108`), which is what showed cos ~0 against the GGUF rows. Env: `CROW_CNQ`, `LAYERDIFF_DIR`, `LAYERDIFF_GGUF` (the UD-Q2_K_XL shard holding `per_layer_token_embd.weight`), `GGUF_PY` (llama.cpp `gguf-py`). Usage: `python3 cnq_ple.py <site> <pos> [arm]`. |
| `flat.py` | The same rows read as the flat block stream (value `160 * r`, the read `ple_row_span` now does) against the GGUF rows. |

`sites.json` maps a site id to `{n_prompt, n_forced, n_total, dump_pos, correct_id, corrupt_id,
last_id}`; the ids files are the prompt ids plus the forced ids of the multi-site probe
(`tools/multisite-corruption-probe.py`).

## Limits

- The crow-nest side of the dump (`decode sitedump`, files `L<ll>-<tag>.f32|i32` per layer) is a
  local instrumentation patch of `engine/src/bin/decode.rs` and `engine/src/gen.rs`. It is not on
  this branch. The patch and every dump and log of the 2026-09-23 run stay in the measurement
  worktree (`decode_out/meas-0923/layerdiff/`, untracked).
- The comparison is against llama.cpp's own quantization (UD-Q2_K_XL), not against BF16. Agreement
  means "the two engines compute the same function up to their different weight quantizations",
  not "both are right".

## The measurement of record (2026-09-23)

Sites `mat44-a149` (103,559 tokens) and `N33-a131` (98,015 tokens), crow-nest with the dense BF16
overlay, pinned 50 GiB WC, `CROW_GRAPH=0`, llama.cpp `-c 200000 -ctk q8_0 -ctv q8_0 -ncmoe 31`:

- Layer 0 matches (residual cos >= 0.9985). Layer 1, the PLE layer, splits: gathered PLE embedding
  cos -0.01 to the GGUF row, PLE contribution cos -0.03 / -0.04, residual cos 0.23 / 0.21, and it never
  recovers.
- With the flat read the gathered embedding is at cos 0.993 and the layer-1 residual at 0.997-0.999.
  Margins (logprob correct - corrupt, nats) move from -0.81 (mat44) / -7.77 (N33) to +12.53 /
  +12.71 (`compare-dense.json` -> `compare-dense-flat.json`). llama.cpp reads +14.60 / +14.16 in
  the same dumps (`ldump`), and +14.25 / +13.37 through llama-server in the multi-site probe.
- The final layer's residual cos rises from 0.74 / 0.54 to 0.91 / 0.92 (mat44 / N33); the remaining gap
  is not attributed (the two weight quantizations differ).
