# HANDOFF — the Crow problem — 2026-09-22 afternoon

Supersedes HANDOFF-2026-09-21-crow-problem.md (read that one only for the
audit trail). State as of this handoff, everything verified on the machine.

## 0. The one goal

Crow must produce **substantially better output quality than llama.cpp**.
llama's precedence is SETTLED — never re-test it. robin's current benchmark:
"Crow must build the diorama from 0 to 100 BY ITSELF" — the assistant must
NOT build it for him. Our job is making Crow CAPABLE of it.

## 1. What the two failed diorama sessions established (read before anything)

- 2026-09-21: 476 req / 46.2M prompt tok / 2.5 h. 2026-09-22: 763 req / 2 h,
  prompts to 174k. Both aborted by robin, zero rendered deliverable.
- Root causes are now TICKETED — do not re-diagnose:
  - **crow#212** — the offline single-file wall: file:// blocks ES-module
    imports (Chromium rule, in the session console verbatim); vendored
    three.js r0.180.0 is a SPLIT build; Crow didn't know and had no bundler
    capability although esbuild 0.25.5 sits in the deno cache since 09-19.
    Fix: knowledge in tool descriptions + build_bundle + dead-end brake hook.
  - **crow#213** — render_page: software-WebGL chromium ran away to 54 GiB
    (froze robin's machine, hard restart), blank 4.6 KB captures the model
    graded blindly, render process outlived the tool call. Fix: MemoryMax
    scope, hard kill by handle, degenerate-capture detection, GPU-when-free
    with swiftshader fallback (crow_core.py:7828-7850 is the anchor).
  - **crow#202** (commented, still open) — the goal-mode brake both sessions
    measured the absence of. Same-error-class counting per step.
  - **crow-nest#91** (commented, still open) — generation corruption:
    fresh instances both days (`33`, `120+1`, `Float332…`, empty read_image
    path, 46+32 dropped-bytes events). Transport re-exonerated (tokenizer
    roundtrips, forward-only emission, split-invariant parser).

## 2. Work order for the new session (robin's ticket set)

1. **crow#213** — render_page hardening. Pure crow repo, testable with the
   existing suite (`cli/test_crow_core.py`, run capped), no engine needed.
   This unblocks ALL visual verification work afterwards.
2. **crow#212** — bundler capability + file:// knowledge. Includes finding
   esbuild (`~/.cache/deno/` holds esbuild-linux-x64 0.25.5 since 09-19;
   node/npm 11.19 installed; `npx --offline esbuild` may resolve from npm
   cache) and a `build_bundle`-shaped tool or run_command guidance.
3. **crow#202** — the brake. Has two sessions of numbers in the comment.
4. Then robin replays the diorama THROUGH CROW (his test, GUI, not ours).

## 3. The #91 arms measurement — STILL NOT RUN (be careful here)

- `tools/corruption-probe.py` + `tools/corruption-arms.sh` (commit e6db340)
  are ready. Arms on disk: `converter/layer91-*.cnq` (6 files).
- **The one launch attempt FAILED SILENTLY**: `nohup systemd-run … python3
  tools/corruption-probe.py …` from `~` — systemd-run does not inherit the
  caller's cwd, the relative path resolved to `~/tools/…` = ENOENT, and
  nothing retried. No baseline number exists. **Use absolute paths or
  `--working-directory=`** (this is the same trap the fleet tests documented).
- Needs: ~45 GiB free RAM (MemAvailable; watch the lazy drain after a serve
  ends), GPU free, ~40 min for 5 arms. Decision rule in the 09-21 handoff §3
  and in the crow-nest#91 comment.
- Sequence with robin: he cancelled infra runs twice when he wanted problem
  work instead — ASK or show progress, don't run it silently in background.

## 4. Machine state at handoff time

- serve on 8099 was ended by robin (~13:5x); GPU ~0.8 GB used. RAM was still
  draining container pages (4.1 GiB avail at handoff) — wait for ≥45 GiB
  before any serve boot.
- **Boot: STILL Windows-first** (`BootOrder: 0000,0002,0003`). robin ran the
  efibootmgr fix once (bash_history line 426) but a Windows repair after it
  rewrote the order again. `~/repair-boot.sh` is written and waiting (backup
  + Limine timeout/menu restore + Windows entry + BootOrder); it needs his
  `sudo bash ~/repair-boot.sh`. The permanent variant (bootmgfw.efi swap) is
  in the script's tail note / ticket #212 era messages — his call.
- `/tmp` gets wiped repeatedly (fleet-monitor state now also mirrored at
  `decode_out/fleet-monitor-state.json`).
- ESP mounted at /boot AND /mnt/esp, root-only.

## 5. Rules paid for — keep every one

- NEVER intervene in robin's running processes (no chmod/kill on his apps).
  The 09-21 chmod-x on the live zcode binary crashed his session → hard
  restart. Balloon runs: vetoed. Chromium runaways: kill only by exact pid
  and only when clearly ours.
- sudo is not available to the agent. Privileged steps = one ready-made
  command/script for HIS terminal.
- Live acceptance through the Crow GUI only. Push is robin-only (crow-nest
  @ e6db340, crow @ 260cc87, both unpushed; #207 fixes live-installed via
  install.sh).
- English for all generated content. Measured facts only in tickets.
- Don't touch the diorama deliverable — Crow must do it 0-100 itself.
- "Chrome installieren?" — answered no: same engine, same file:// rule; the
  fixes are #213's (ceiling/GPU/blank-detection), not another binary.

## 6. Repos / artifacts

- crow-nest @ `e6db340` local: probe + arms runner committed. Untracked:
  layer91 arms, both HANDOFF files, older analysis files.
- crow @ `260cc87` local: #207 (bounds+capture cap, LIVE-ACCEPTED),
  min_p manifest fix, acceptance docs.
- Fleet: 22 subagents DONE, state durable-mirrored; monitor cron may still
  fire the standing prompt — answer it in ≤3 lines when quiet.
