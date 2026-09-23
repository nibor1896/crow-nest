# HANDOFF 2026-09-21 (afternoon) — wave 2 closed, the fleet's second life

Written by the orchestrator after closing wave 2. Everything local, nothing pushed. `main` is 11 commits ahead of origin.

## State of the board (issues #83–#97)

- **Implemented + accepted + committed:** #83, #84 (192c099), #85, #86, #92, #96, #97/F1 (59ceb42), #87 phase 1 (d8b66a0), #88 (22a274f), #89 + all leftover probes P1b/P2/P3 (08ae3c8, 59ceb42), #90 (74d6a3e), #91 phase 1 (705be35), #94 (5662fea).
- **Open with substance remaining:** #87 phase 2 (fresh record at the full operating row — needs engine + llama shard 1 + Crow), #93 (grammar/constrained decoding), #95 (MTP speculative), #91 measurements (arms built, measurement post-fresh-boot), #90 engine arm (RAM).
- Test suite: **305/0**. Clippy 1505 (same classes). All three doc guards green. Env table 106 rows, checker green.

## The gate, honestly

- **08:11 run: parity8, parity512, p8tf ALL GREEN with the exact shas of record at full tier** (bceba6ff…, 83872347…, 3bb3e69e…) — the byte-identity contract held through EVERY wave-1+wave-2 engine change. clippy + guards green.
- **08:21 run: cargo test 305/0 GREEN** (the gate's test item now carries the nvrtc library path — the #96 ptx test needs it).
- **run32 is the one outstanding item**: blocked by the day's RAM baseline (balloon ceiling 42.9 GiB < the ~49 the -M default tier wants; this session + system hold ~19 GiB). One command on a fresh boot/session: `bash tools/gate-linux.sh` — expect ALL GREEN. The gate now pool-recovers between GPU items with a 240 s balloon budget.

## The incident record (why the fleet died twice)

1. The stopstr test helper `cut()` looped without progress on multi-byte characters at piece size 1 → unbounded piece-list growth → 28–30 GB test processes → OOM → **ZCode died with the fleet inside** (first death, robin rebooted 07:41). The relaunched fleet hit the same test again (second death, orchestrator caught and killed the processes this time). Fixed + tombstone comment in `engine/src/stopstr.rs`.
2. The `0f3E000000`-is-0.0625f test hex was wrong (it is 0.125f); the PTX was right.
3. Two degenerate test fixtures (t=0 angles, ~1e-5 rad pair-31 angles) asserted distinguishability where f32 cosines coincide — fixed with angle-magnitude guards.

## robin's checklist (fresh boot or after closing this session)

1. `bash tools/gate-linux.sh` — expect ALL GREEN (closes run32; the only item that never had a full-tier pass today).
2. Relaunch the #90 engine arm: `nohup tools/oracle_longctx_engine_arm.sh > decode_out/oracle-longctx/engine-arm.log 2>&1 & echo $! > /tmp/fleet-monitor/90-oracle.pids`
3. Optional for #87 phase 2 + #90's llama arm: re-download GGUF shard 1 (~25 GB).
4. Live acceptance per unit: `docs/acceptance/issue-{83,84,85,86,87,88,89,89-followup,90,91-phase1,92,94,96}.md` — curl commands, expected provenance lines, parity A/B (`CROW_SAMPLE_HOST=1`), probe re-runs.

## The quality story, one paragraph

The engine's sampler surface is now at llama.cpp default-chain parity (min_p, windowed penalties, DRY, top-nσ, typical, XTC, mirostat v2, stop strings, logit_bias — every member, all neutral by default, goldens byte-identical). The numerics are closed: formulas acquitted three-way (#89), KV excluded (#88), GDN prefill measured at 6.4–7.1e-7 against the HF chunked reference with flat state error (P2), torch-CUDA tie order matches the engine rule (P1b). What remains as the quality lever is the weight quantization (#91 arms, measured by the #90 instrument) and the operating point itself — the record at the real row (#87 phase 2) is the number that tells robin whether the German non-words moved.
