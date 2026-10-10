#188 cache-policy simulation, 2026-10-10 (tools/glm_tier_sim.py arena, docs/glm-tier-simulation.md section 9).

Inputs
- route.log, decode.log, decode.json, env.txt: glm5_run on the quick decode prompt (runs/glm53-flash/quick/quick.sh,
  -n 128, --cold) with the operating set minus guesses (arm2.env without CROW_GLM_PREFETCH / _SIDE, + CROW_GLM_RT2=1),
  worktree glm-cache (28f459a + the CROW_GLM_ROUTE_LOG hook), CROW_GLM_ROUTE_LOG=route.log. Its counters equal
  q7-rt2-nopf (32.07 NVMe reads per token, VRAM 36.1 % / pinned 54.4 %).
- held-out todo-1006 of decode_out/glm-step8 (FP8-originals routing, capture-ids for the calibration files), warm file
  runs/glm53-flash/arena-warm/glm53-cal4-decode-counts.json.

Outputs
- arena.txt/.json: every policy on both workloads at V 2220 / P 4956 (the decode slots of the measured run: 1848 base
  - 24 ring + 9 of 13 elastic chunks x 44; 118 x 42 pinned), half-life 64, margin 1.
- arena-freq-margin*.{txt,json}, arena-halflife*.{txt,json}: freq sweeps. arena-vram*.{txt,json}: capacity sweep on the
  held-out (today, freq, MIN).

Command (from the worktree root; B = C:/Users/robin/dev/crow-nest/decode_out/glm-step8):
  python -I tools/glm_tier_sim.py arena --corpus $B/corpus/corpus.json --runs $B/runs --capture $B/capture-ids \
    --warm runs/glm53-flash/arena-warm/glm53-cal4-decode-counts.json --route-log <this dir>/route.log \
    --decode-log <this dir>/decode.log --boot 1848:24:13:44:4 --json <this dir>/arena.json
