## GPU acceptance (2026-10-11, robin's GPU go)

**Conditions:**
- Hardware and model: RTX 5090, Windows 11. Container `converter/GLM-5.3-Flash-MUL1K3.cnq`, `runs/glm53-flash/quick/arm4.env`, `serve --port 8099`, `CROW_CONTEXT` unset (200,000), MTP off.
- One engine at a time under `<home>\dev\.gpu-lock`. Nothing else on the GPU, Crow closed.
- Binaries:
  - **after** = local main `2a18fbc` (#215 + #218 + #219 + #217 + #216, new switches off).
  - **before** = `6944330` (the step-0 commit), built in a temporary worktree.
- Boots: one per (prompt, binary), 21-22 s to `/health`. Free VRAM before each boot: 0.94-1.09 GiB used by others. Available RAM 52-53 GiB.
- Prompts:
  - P1: #214's quick question, 57 ids, `max_tokens` 512.
  - P2: #213's request 2, `docs/glm5-model.md[0:12000]` at `298092b` + its question, 3,985 ids, `max_tokens` 512.
  - P3: G5 t1 request 1, Crow's body rebuilt byte for byte (27,159 B as in the G5 log), 6,438 ids, `max_tokens` 1024.
  - Thinking on for all three. Seed 1408639702 for every sampled arm.
- Arms:
  - G: `temperature` 0.
  - H: G + `logit_bias {"0":0}` (before binary).
  - S: temperature 1, top_p 0.95, top_k 20, min_p 0.01, seed, + `logit_bias {"0":0}` (host draw, after binary).
  - D: the S body without the bias (device draw).
  - F: G + `crow_force_ids` = the ids of S's capture.
- Order per boot:
  - before: Gcap, G H H G G H.
  - after: Gcap Scap Dcap, then G S F D D F S G G S F D.
- A *cap* request is the arm + `logprobs: true` + `crow_force_ids: []`, so that it returns `crow_id` per generated id. Caps are not timed.
- Every timed request is WARM on its prompt (`prefill 0 of N`).
- ms/token = `predicted_ms / predicted_n`. NVMe/token = the delta of the `[chat] counters` nvme accesses / predicted_n.
- Raw: `runs/glm53-flash/accept-20261011/215/` (untracked: `rows.jsonl`, the serve logs, `analysis.txt`). Scripts: `run215.py`, `harness.py`, `bodies.py`.

**ms/token, median of 3 (the runs in brackets):**

| prompt | G before | H before | G after | S (host) | F | D (device) |
|---|---|---|---|---|---|---|
| P1 | 53.313 [53.31 53.01 53.64] | 54.024 [54.06 53.49 54.02] | 52.154 [52.15 53.08 50.73] | 53.205 [53.16 53.82 53.21] | 51.908 [52.39 51.91 51.50] | 53.087 [53.72 52.48 53.09] |
| P2 | 68.733 [68.73 68.94 68.02] | 69.677 [69.68 69.52 69.86] | 68.272 [68.66 67.65 68.27] | 73.148 [73.74 72.86 73.15] | 71.668 [71.65 71.91 71.67] | 71.478 [70.00 72.51 71.48] |
| P3 | 68.168 [68.17 66.97 69.90] | 71.330 [73.03 71.33 70.15] | 71.035 [71.04 70.38 75.79] | 69.357 [72.58 69.36 67.30] | 67.189 [68.05 65.77 67.19] | 67.093 [64.58 67.09 68.81] |

NVMe reads per decode token, medians:
- P1: G 8.3 / H 8.1 / G' 8.0 / S 8.1 / F 8.0 / D 8.2
- P2: 14.4 / 14.4 / 14.3 / 16.3 / 15.7 / 16.0
- P3: 13.7 / 14.3 / 14.7 / 13.5 / 13.0 / 13.0

On P3, the generated length varies by arm and run: 510-1,024 tokens, ending at a tool call or at `length`.

**Finding that governs criteria 1 and 3: free decoding is not reproducible run to run on either binary.**
- The texts of the 3 timed G runs differ from each other, and from Gcap, on the same boot and the same binary. On P1 before, the first differing chars are at 387-958 of ~2,150.
- The same holds for S and D: 3 distinct texts each on every prompt, none equal to its capture.
- Only F, the forced arm, repeats exactly: 3/3 equal to Scap's text on every prompt.
- So the logits rows themselves differ between runs, and every id comparison between two free runs tests this nondeterminism, not the sampler.

**Criteria as written:**
1. **Ids. FAIL as written.**
   - Dcap vs Scap first differ at id 31 (P1), 387 (P2) and 79 (P3).
   - Same seed twice: D's 3 runs + Dcap give 4 distinct texts.
   - Per the ticket's rule, a differing id calls for the frequency test (5 recorded logits rows, 200,000 draws, TV ≤ 0.005, KL ≤ 1e-3). It was **not run**: it needs recorded rows and a host harness, which is a code change outside this block.
   - At the mechanism level, the ignored GPU test `glm5_devsampler_gpu_draws_are_the_host_chain` passed. Device draws equal `Sampler::sample` + `observe` on the same rows, for 3 profiles (GLM row, window, hot), the same seed twice gives the same ids, and parked gives the argmax (`gputests/215.log`, 1 passed).
2. **Speed. D ≤ F × 1.01: P1 FAIL, P2 pass, P3 pass. D < S: pass on all three.**
   - P1: 53.087 vs 52.427, FAIL. P2: 71.478 ≤ 72.385, pass. P3: 67.093 ≤ 67.861, pass.
   - D < S: 53.087 < 53.205, 71.478 < 73.148, 67.093 < 69.357.
   - S − D = 0.12 / 1.67 / 2.26 ms/token. Step 0's H − G (on `6944330`) = 0.71 / 0.94 / 3.16 ms/token.
   - Note: F replays Scap's trajectory, while D follows its own (see the finding). On P1, D read 8.2 NVMe/token against F's 8.0.
3. **Greedy unchanged:**
   - Ids identical: FAIL as written. Gcap after vs before first differ at id 28 / 302 / 129. G is not self-reproducible on either binary, so this does not separate the fix from the nondeterminism.
   - ms/token within ±2 %: P1 0.978 (−2.2 %), FAIL; P2 0.993, pass; P3 1.042 (+4.2 %), FAIL. On P3 the greedy runs generated 510-1,024 tokens, so the trajectories differ.
4. **Grammar:**
   - Live, after binary, P3: 15 requests, all `malformed 0`. 11 of them ended in tool calls (1-2 calls each), and 4 at `length` with no call opened. The grammar checked every request: 0 refused, 0 redrawn.
   - On the before binary, 2 of 7 G/H runs were cut at `max_tokens` inside a call (`malformed 1`, the 1,024 cap).
   - The unit test with the forced refusal and re-book is host-side and was green in the implementation run. Not re-run here.
5. **Route line: pass.**
   - Boot: `[serve] glm5_next: device sampler on (...)` and `[budget] device sampler (#215): 501556 B ... held from boot`.
   - D: `[chat] sampling on the device: ...`.
   - S: `[chat] glm5_next draws on the HOST sampler (#215): logit_bias, 1 entries (#86)` + `sampling on the HOST`.
   - MTP, MAX_BATCH and DRY were not exercised.

**Summary:**
- The device sampler is on and routed as specified, and it is never slower than the host draw (S − D 0.1-2.3 ms/token).
- The id-equality criteria (1, 3) cannot pass in serve while free decoding is itself nondeterministic. The ±1 % / ±2 % speed criteria fail on P1 (D vs F), and on P1 and P3 (G before/after), at a run-to-run spread of up to ±3 ms/token within one arm.
- No threshold was changed.
