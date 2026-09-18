# Ten-task gate — fixation (spec §5.2)

Fixed 2026-09-03, before the #11 standing series starts, from Crow's real
workload. Seed set: crow-lab `runs/2026-09-01-tasks` (t1–t6, A/B prompt
variants). Each task runs IDENTICALLY on both engines (crow-nest engine vs
llama.cpp baseline, same session, interleaved) and the answers are compared
for CORRECTNESS — the gate protects against "faster but worse".

| # | id | class | seed | shape |
|---|----|-------|------|-------|
| 1 | t1-read | Repo lesen + zusammenfassen | t1-read-A | code file → summary |
| 2 | t2-write | Kleine Funktion schreiben | t2-write-A | spec → code |
| 3 | t3-debug | Fehler in Code finden | t3-debug-A | snippet → root cause |
| 4 | t4-prose | Technische Prosa | t4-prose-A | topic → text |
| 5 | t5-agent | Mehrstufiger Arbeitsauftrag | t5-agent-A | task → plan + steps |
| 6 | t6-reason | Rechnen/logisches Schließen | t6-reason-B | problem → answer |
| 7 | t1b-read-lang | Lesen am langen Kontext | neu (Crow-Realität: große Files) | file > 8k tokens → Fragen |
| 8 | t3b-debug-syn | Debug mit Synthese aus zwei Stellen | neu | two snippets → bug |
| 9 | t2b-write-refactor | Refactor-Auftrag | neu | code → refactored code |
| 10 | t6b-reason-multi | Mehrschritt-Kette | neu | chained problem → answer |

Measurement discipline (robin, 2026-09-03 — binding for the run harness):
- **Positions rotate per ten-task run AND per arm.** Every run r uses a
  rotated task order (shift by 3r, reversed on odd runs) and the arm that
  moves first swaps (even run: crow first, odd run: llama first). Positions
  in a series must not favour either arm — llama.cpp's prompt cache makes
  later turns cheaper, and the crow engine warms its caches symmetrically;
  both arms see the same rotated order (paired), only the rotation changes
  per run.
- **Cold prefill per start is paid and discarded.** llama-server pays one
  cold prefill after every start/reboot (model mmap + empty prompt cache) —
  the first turn after a (re)start is always slower. The harness sends a
  throwaway request per arm after every start and discards it; restarts
  mid-series re-warm and are recorded as events in the run metadata.
  Same discipline for the crow-nest arm (one discarded throwaway after
  engine load).

Rules:
- Greedy/temperature 0 on both engines (determinism over realism for the gate).
- Prompts live in `decode_out/ten-tasks.json`; expected-answer notes in
  `docs/ten-task-expected.md` (the ten-task record keeps both engines' raw answers). The
  `parity-prompts.json` / `parity-expected.md` names this line carried until 2026-09-17 never
  existed in the tree.
- A task PASSES if crow-nest's answer is materially correct AND consistent
  with the llama.cpp answer where both should agree (facts, code semantics);
  wording may differ.
- The gate runs at the 200k-floor operating point for the long-context tasks
  (7) and at natural prompt lengths for the rest; operating points are recorded
  per run (spec §0.5).
- Misses are measured results: a failed task re-cuts the suspect stage
  (quant: CNQ4.5-C lever for attention — p16 finding; kernel: numerics gate).
- **A dead oracle child costs the TASK, not the phase** (issue #65, 2026-09-18). The two
  python children of the oracle venv — `tools/tokenize_ids.py --chat` per task and
  `tools/detokenize_ids.py` per phase — get three attempts with a 2 s and then a 5 s pause,
  because three times (chains 19f and 19g, 2026-09-13, and the 62b pairs preflight) one
  tokenize child died at task 5 with a non-zero exit and an EMPTY stderr and every retry on
  the identical input was green. It stays fail-closed: after the third attempt the phase
  fails and records nothing for that task. A retry that succeeded is part of the run record
  (`oracle_retries` on that task's row, with the exit code and the captured stderr), so a
  number measured after a retry is never quoted as if nothing had happened
  (`docs/architecture.md` 8.9).

Provenance: t1–t6 = crow-lab 2026-09-01 task series (robin's Crow workload
sample); t7–t10 added 2026-09-03 to cover long-context reading and
multi-step synthesis that the seed set does not exercise. The frozen seed texts (t1–t6) are EXTRACTED, read-only, from
crow-lab `runs/2026-09-01-tasks/*.txt` into
`docs/ten-task-prompts-crowlab.json` (hashes verified against
`tasks-manifest.json` by crow-lab's own freeze discipline).
