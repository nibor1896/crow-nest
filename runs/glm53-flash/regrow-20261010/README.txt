#188 elastic-chunk regrowth (CROW_GLM_ARENA_REGROW), 2026-10-10, branch glm-regrow (from glm-cache b573f4c).

smoke/off, smoke/on: glm5_run on the quick decode prompt (runs/glm53-flash/quick/quick.sh decode line, crow-nest-wt-int),
  -n 32 --cold --reps 1, operating set arm2.env + CROW_GLM_RT2=1 + CROW_GLM_ARENA_FREQ=1, and CROW_GLM_ARENA_REGROW=0 / 1.
  Correctness and slot counts only (not a speed measurement). Line "glm5_run arena after rep 1" and reps_detail[0].arena:
  off: 4 chunks handed back at 1.43 GiB free, 8 of 11 elastic chunks x 45 live, 2,226 VRAM slots in decode;
  on: 4 handed back at 1.41 GiB free, regrown 4 at 3.01 GiB free (floor 1.41 GiB), 11 of 11 live, 2,361 VRAM slots,
      refilled 180 records (0 from the NVMe), free VRAM after the run 1.45 GiB. Ids equal.

arena-quick-boot-*.{txt,json}: tools/glm_tier_sim.py arena on the quick decode routing of runs/glm53-flash/cache-sim-20261010
  (127 decode tokens, no guesses), boot 1848:24:13:44:4 (today: 4 chunks down) and 1848:24:13:44:4:1 (regrown + refilled):
  python -I tools/glm_tier_sim.py arena --warm <crow-nest-wt-int>/runs/glm53-flash/arena-warm/glm53-cal4-decode-counts.json \
    --route-log runs/glm53-flash/cache-sim-20261010/route.log --decode-log runs/glm53-flash/cache-sim-20261010/decode.log \
    --boot 1848:24:13:44:4[:1] --policies today,freq --json <out>.json
arena-heldout-vram2396.{txt,json}: the held-out todo-1006 at 2,396 VRAM slots (2,220 + the 176 regrown), same command with
  --corpus/--runs/--capture of decode_out/glm-step8 and --vram 2396 instead of the route log.
