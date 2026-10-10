#188 round m cache-policy simulation, 2026-10-10 (tools/glm_tier_sim.py arena, docs/glm-tier-simulation.md section 9.1).

Question: how far is the operating set's policy (CROW_GLM_ARENA_FREQ=1) from Belady MIN at today's capacity, and which
practical policy closes part of the gap.

Inputs
- route.log of ../cache-sim-20261010/: the quick decode prompt's own routing (57-id question, 127 decode tokens, one
  prompt call), replayed with today's boot: 1848 base - 24 ring + 12 x 44 elastic (regrown, lazy) + 128 lent = 2,480
  VRAM slots, 4,956 pinned = 7,436 slots (the round-l run had 2,480 VRAM slots, crow-nest-wt-int
  runs/glm53-flash/profile-20261010l-decode/plain.log).
- held-out todo-1006 of decode_out/glm-step8 (26,599 decode tokens, 32 prompt runs cut into 55 prompt calls of at most
  185 positions, the decode-time prompt chunk of the round-l run), V 2,352 + 128 lent + 4,956 pinned = 7,436 slots.
- warm file runs/glm53-flash/arena-warm/glm53-cal4-decode-counts.json; half-life 64, margin 1, prior 46.2 tokens,
  lent 128, lazy refill (the operating set).

Outputs
- route.txt/.json, held.txt/.json: every policy (today, promote, lfu, tinylfu, noadmit, freq, fprompt, fpadmit, fguard,
  lru, arc, s3fifo, min) on one workload each.
- margin{0,0.5,2}, halflife{32,128}: freq and fprompt with that margin / half-life, both workloads per file.
- pw1, pw4: fprompt (and fpadmit) at prompt weight 1 / 4; guess0.5: fguard at guess precision 0.5.

Commands (from the worktree root; B = C:/Users/robin/dev/crow-nest/decode_out/glm-step8,
W = runs/glm53-flash/arena-warm/glm53-cal4-decode-counts.json, C = runs/glm53-flash/cache-sim-20261010):
  python -I tools/glm_tier_sim.py arena --warm $W --route-log $C/route.log --boot 1848:24:12:44:4:1 --lend 128 --lazy \
    --json <this dir>/route.json
  python -I tools/glm_tier_sim.py arena --corpus $B/corpus/corpus.json --runs $B/runs --capture $B/capture-ids --warm $W \
    --vram 2352 --lend 128 --lazy --json <this dir>/held.json
  sweeps: both workloads in one call (--corpus ... --vram 2352 --route-log ... --boot ...) with --policies freq,fprompt and
    --margin X / --halflife X; --policies fprompt,fpadmit --prompt-weight X; --policies fguard --guess-precision 0.5
