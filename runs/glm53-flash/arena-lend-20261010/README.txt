#188 T (CROW_GLM_ARENA_LAZY_REFILL) and the lent prefill staging set (CROW_GLM_ARENA_STAGE_LEND), 2026-10-10,
branch glm-arena2 (from glm-integration 7ba2864).

smoke/{off,lazy,lend,lend-lazy}.{log,json}: glm5_run on the quick decode prompt, -n 32 --cold --reps 1, operating set
  crow-nest-wt-int runs/glm53-flash/quick/arm2.env + CROW_GLM_RT2=1 CROW_GLM_ARENA_FREQ=1 CROW_GLM_ARENA_REGROW=1
  CROW_GLM_EMBED_GATHER=1 CROW_GLM_RT2_TABLE_SM=1, each switch 0/1. Correctness and slot counts only, not a speed
  measurement. Ids equal in all four arms (32 ids). reps_detail[0].arena / "glm5_run arena after rep 1":
  off        2,361 VRAM slots in decode (11 of 11 elastic chunks x 45; 4 regrown, refilled 180)
  lazy       2,361 slots, refilled 0, 180 slots left to admission
  lend       2,489 slots (+128 lent), refilled 308 (180 regrown + 128 lent)
  lend-lazy  2,489 slots, refilled 0, 308 left to admission
  stage_buffers_vram_bytes 0 in decode in every arm (the CROW_GLM_ARENA_STAGE_GB buffers exist only while a staged
  forward is open; this 57-token prompt was not staged: 456 picks < 512). prefill_staging_slots 128.
  decode.counters.hits_per_token.vram (32 tokens, placement only): off 155.8, lazy 159.9, lend 160.6, lend-lazy 166.3;
  nvme_demand_reads_per_token: 14.03, 14.03, 13.45, 13.45.

quick-*.{txt,json}, heldout-*.{txt,json}: tools/glm_tier_sim.py arena, policies today,freq (no guesses):
  W=runs/glm53-flash/arena-warm/glm53-cal4-decode-counts.json, C=runs/glm53-flash/cache-sim-20261010,
  B=C:/Users/robin/dev/crow-nest/decode_out/glm-step8
  python -I tools/glm_tier_sim.py arena --warm $W --route-log $C/route.log --boot 1848:24:13:44:5:1 \
    --policies today,freq [--lazy] [--lend 128] --json <out>.json
  python -I tools/glm_tier_sim.py arena --warm $W --corpus $B/corpus/corpus.json --runs $B/runs --capture $B/capture-ids \
    --vram 2396 --policies today,freq [--lazy] [--lend 128] --json <out>.json
  boot 1848:24:13:44:5:1 = the decode slots of profile-20261010k-decode (13 x 44 elastic, 5 handed back and regrown).
  freq rows (demand NVMe reads / VRAM hits per token / ms per token, cost model 1.7 ms per demand read):
    quick   eager 27.76 / 184.7 / 67.1   lazy 27.76 / 186.1 / 67.1   lend 26.65 / 189.6 / 64.8   lend+lazy 26.65 / 191.1 / 64.8
    heldout eager 34.35 / 170.6 / 77.9   lazy 34.35 / 170.6 / 77.9   lend 32.61 / 175.6 / 74.4   lend+lazy 32.61 / 175.8 / 74.5
  The refill stall (eager: 66 ms per prompt in profile k) is not in the cost model.
