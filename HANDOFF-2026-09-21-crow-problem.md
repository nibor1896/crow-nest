# HANDOFF — the Crow quality problem — 2026-09-21 evening

For a fresh session. Written after the diorama live-test failure and the
session crash. Read this whole file before touching anything.

## 1. The problem, in robin's words

Crow must produce **substantially better output quality than llama.cpp**.
That is the only goal. Everything else (gates, byte-identity, tickets) is
instrumentation. llama.cpp's superiority is ESTABLISHED (blind grading,
2026-09-21 morning: Linux record, 16k deficit = thinking budget, crow draws
level at 32k) — do NOT re-test llama. Beat it here.

The visible symptom: every long Crow session dies in a swamp — corrupted
code (`[0,1,2,0,2,33]` for `[0,1,2,0,2,3]`, `1..05`, `0x46566b,,`,
`Float332BufferAttribute`, `120 + 1` for `16`, `df5fe00` for `df5fe09`,
`Cargo.tomi` for `Cargo.toml`, `pattern_2:"placeholder"` garbage tool args),
46 "bytes dropped" engine-log events in one afternoon, 2.5h / 476 requests /
46.2M prompt tokens burned on one task with zero deliverable rendered.

## 2. Attribution state (proven, do not re-litigate)

- **Engine code EXONERATED** (full audit 2026-09-21): tokenizer roundtrips
  green (ASCII+UTF-8 tests in tokenizer.rs), stream emission is forward-only
  with UTF-8 boundary guards (serve.rs next_delta), tool args accumulation is
  append-only, ToolStream parser is byte-accounted and split-invariant by
  test (toolcall.rs:988). The corruption is GENERATED, not transported.
- **Sampling is correct**: min_p 0.01 live end-to-end (manifest fix, LIVE-
  ACCEPTED today), full llama.cpp chain implemented (#83-#92), parity
  byte-identical (gate ALL GREEN at df5fe09).
- **Prime suspect: NVFP4 weight quantization (#91)**. Evidence: corruption
  persists with correct sampling; #89 acquitted the formulas; #88 excluded
  KV; GDN prefill sits on the torch floor (6.4e-7); the corruption signature
  is trailing-digit/token damage in repetitive code — the quantization
  residual class. THE DECISIVE MEASUREMENT IS BUILT AND COMMITTED (see §3).
- Secondary hypothesis covered by the same instrument: stream-vs-nonstream
  differential (probe has both modes). If corruption appears only in stream
  mode after all, reopen the transport audit — but code reading says no.

## 3. Ready to run — the #91 arms measurement (FIRST THING NEXT SESSION)

- `tools/corruption-probe.py` (committed e6db340): seeded hex-literal
  reproduction, 8 rounds x 40 literals, per-line/per-char error rates,
  stream and non-stream modes, JSON output.
- `tools/corruption-arms.sh` (committed e6db340): boots serve 5x —
  baseline, attn placebo, attn-v-out BF16 (all layers), ffn-down rule
  (llama.cpp use_more_bits layer set), ffn-down all — runs the probe on
  each, prints one summary table. Results land in
  `decode_out/corruption-arms/`.
- Arms on disk (untracked): `converter/layer91-{attn-v-out,ffn-down-rule,
  ffn-down-all}-{originals,control}.cnq` (originals = BF16 overlays,
  control = byte-identical placebo). Applied via `CROW_CNQ_OVERLAY=<file>`.
- **Needs ~45 GiB free RAM** (check `MemAvailable`; serve -M pins 45 GiB).
- Decision rule: an arm that drops line_error_rate below baseline AND
  placebo confirms quantization → scale the winning overlay shape to the
  worst-K layers (machinery exists: layer_rule_overlay.rs), re-measure,
  then the corruption class should shrink. If NO arm moves the number,
  quantization is acquitted too → next suspect is the sampling-adjacent
  logits path (logit storage/order, #90's instrument at the real row).
- robin must NOT be asked to run this; it runs from a terminal here. But:
  he cancels infrastructure runs when he wants problem-work instead —
  sequence it WITH visible progress or ask him first.

## 4. Crow client fixes queued (the half he watches every day)

1. **#202 goal-mode brake** (open ticket, describes today exactly: 300 turns
   on one step, 105 identical nudges). After N identical-failure rounds on
   one goal step, force replan/escalate instead of grinding. This is the
   "geisteskranke Runden" killer.
2. **render_page hardening** (new ticket needed): today's render chromium
   ran away to 54 GiB RAM (software WebGL on a 2 MB three.js page) and
   FROZE THE MACHINE (forced the hard restart). Also: byte-identical
   capture warnings (canvas not reaching screenshot), final render no
   screenshot + PHONE_REGISTRATION_ERROR console garbage. Needs: memory
   cap + hard timeout + fresh profile per run + kill guarantee.
3. **#207 is FIXED and LIVE-ACCEPTED** (search bounds, capture cap,
   unknown-args note; a045caf + d22f3a2 + docs 260cc87) — local only.

## 5. Machine state — read before ANY action

- **Boot**: Windows repair rewrote BootOrder to Windows-first; robin's
  selection menu (Limine, timeout>0) stopped showing. Repair script at
  `~/repair-boot.sh` (backup + timeout fix + windows entry + BootOrder).
  May or may not have been run — CHECK `efibootmgr | grep BootOrder` first.
  ESP is mounted at /boot AND /mnt/esp, root-only (fmask=0077).
- **NEVER intervene in robin's running processes** (no chmod on running
  apps, no killing his desktop/compositor/zygotes). 2026-09-21: a chmod -x
  on the running zcode binary during a freeze window crashed his session →
  hard reboot → this handoff exists. Balloon runs were also vetoed by him.
- **serve leaves a uvm wedge after hard kills**: lazy-drains (minutes) or
  stays wedged (reboot only). Check `MemAvailable` >= 45 GiB before serve.
  The render-chromium runaway masquerades as this leak — check for leftover
  `chromium --headless` (crow-render-*) holding RAM before blaming uvm.
- **`sudo` is not available to the agent** (no passwordless). Anything
  privileged goes to robin's terminal as ONE ready-made command/script.
- Fleet: 22 subagents all DONE, state rebuilt at /tmp/fleet-monitor/. The
  30-min monitor cron may still fire the standing prompt.

## 6. Repos

- **crow-nest** @ `e6db340` (local, unpushed; 14+ commits since origin).
  Untracked: converter/layer91-*.cnq (6 files, ~3.4 GB — the arms), this
  handoff, older HANDOFF/IDEA/docs files.
- **crow** @ `260cc87` (local, unpushed): #207 fixes + protocol + min_p
  manifest fix (e7d10a7) + OOM capture cap (d22f3a2).
- **PUSH IS ROBIN-ONLY.** Live acceptance through the Crow GUI only, never
  curl. English for all generated content. Report, then wait.

## 7. Working rules with robin (paid for, keep them)

- He tests through the Crow GUI. Terminal commands for HIS hands get exact
  one-liners, not lectures.
- Do not re-prove established facts (llama better; engine clean). Move the
  open question only.
- No new measurement tooling when he asks for the problem to be solved —
  sequence: run what exists, then fix, then measure the fix.
- When he says "löse das Problem" he means: visible progress on output
  quality or the Crow client's failure modes — not diagnostics of his
  machine.

## 8. Open decisions that were robin's when this session ended

- Run the arms ladder (§3) — his go needed since it occupies the GPU/box.
- #202 + render_page hardening order.
- Push decision for both repos.
- OOMScoreAdjust=-500 for serve (who dies in a global OOM) — one-liner in
  tools/serve-linux.sh, his call.
