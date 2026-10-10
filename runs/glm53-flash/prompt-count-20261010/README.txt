#188 CROW_GLM_ARENA_PROMPT_COUNT smoke, 2026-10-10 22:31-22:49, RTX 5090, real container GLM-5.3-Flash-MUL1K3.cnq,
glm5_run built 22:31:39 from glm-cache2 (3de5eff + the switch), under C:\Users\robin\dev\.gpu-lock (locked.sh).
smoke.sh: the quick.sh decode command (57-id question, --cold --reps 1), operating set (arm2.env + RT2 + the twelve
switches of quick/b5-all12), -n 32 (31 decode tokens). No speed claim; the logs are not committed.

Runs (JSON per arm; NVMe-sourced = demand reads + used guesses per decode token):
- smoke1/off, smoke1/on (CPU lane split): NVMe 36.74 / 36.84, VRAM hits 166.90 / 173.81; VRAM slots 2,489 / 2,444; ids equal.
- smoke/off, smoke/on (split, second pair): NVMe 36.71 / 36.42, VRAM hits 163.55 / 176.39; slots 2,444 / 2,489; ids differ
  from token 28 on. The off run is the odd one: smoke1/off, smoke1/on and smoke/on carry one sequence, smoke/off the other
  (the round-l runs' sequence). Two switch-off runs differ from each other: under split the CPU lane's combos have other
  bits than the GPU's, so where a record lies can flip a near tie; not the switch.
- smoke-nolane1/{off,on}-nolane, smoke/{off,on}-nolane (CROW_GLM_CPU_LANE=0, the GPU reads every record with the same
  bits): NVMe 36.87 / 36.84 and 36.74 / 36.84, VRAM hits 164.03 / 173.84 and 166.97 / 173.94; all four ids equal.
  (smoke-nolane1's off log was overwritten by the second pair; its JSON is the first run's.)
- arena.prompt_count: on = 1 call, 1 fold; off = 0, 0.
Sim on the same 31 tokens (route log of cache-sim-20261010, boot 1848:24:12:44:4:1, lend 128, lazy): freq 36.52 NVMe /
165.4 VRAM hits, fprompt 35.39 / 174.8, MIN 35.32.
