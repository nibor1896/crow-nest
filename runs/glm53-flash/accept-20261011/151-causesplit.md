## G5 row-1 decode gap: cause split (2026-10-11, inside the #215-#219 GPU block)

**Question:** in the G5 Crow session, t1 request 1 (6,438 prompt tokens) decoded at **8.4 tok/s** over 2,187 tokens. In #215's P3 rows the same body decoded at 13.5-15.5 tok/s, capped at 1,024 tokens. Two candidates are testable here:
- the G5 resource sampler (`g5_session.py sampler`, which polls nvidia-smi every 2 s);
- the answer length.

Crow beside serve was not tested; robin rejected running Crow during measurements.

**Conditions:**
- Software: local main `2a18fbc`, `runs/glm53-flash/quick/arm4.env`, every new switch off (#219 / #217 / #216), `CROW_CONTEXT` unset (200,000). RTX 5090, Windows 11.
- Isolation: GPU lock held, nothing else on the GPU, Crow closed. A fresh boot per run (21-22 s to `/health`).
- Request: G5 t1 request 1, rebuilt byte for byte as Crow sent it (27,159 B with `max_tokens` 16384; here `max_tokens` 2,500). That is temperature 1, top_p 0.95, min_p 0.01, top_k 20 from the card, seed 1408639702, thinking max.
- Sampling: device sampler (#215, `sampling on the device`), whereas G5 drew on the host.
- Arms, alternated A B A B (n = 2 each):
  - A: nothing else running.
  - B: a copy of `g5_session.py` run as `sampler` in the run's directory, hidden. It tails that run's serve log and wrote one resource row every 2 s (43 / 48 rows), as in the session.
- Window rates: computed over SSE content/reasoning/tool chunk arrival times. Chunks are 98-99 % of `predicted_n`.
- NVMe/decode token = (`[chat] counters` nvme − the prompt phase's `NVMe 5314 reads`) / generated.
- Raw: `runs/glm53-flash/accept-20261011/causesplit/` (untracked).

| run | generated | decode tok/s (serve) | tok/s chunks 1-1,024 | tok/s chunks 1,025-end | NVMe reads / decode token | prefill ms |
|---|---|---|---|---|---|---|
| A r1 | 2,233 | **14.14** | 15.57 | 12.93 | 14.64 | 7,422.0 |
| B r1 | 1,192 | **15.74** | 16.03 | 12.82 | 11.97 | 7,021.6 |
| A r2 | 1,943 | **14.72** | 15.49 | 13.70 | 13.71 | 6,876.9 |
| B r2 | 1,253 | **14.41** | 14.77 | 12.07 | 14.33 | 6,912.1 |
| G5 row 1 (Crow session, host draw) | 2,187 | **8.4** | - | - | ≈ 14.2 (36,379 cumulative − ~5,314 prompt reads, derived) | 7,263.7 |

**Result:**
- **The resource sampler is not the cause.** B ran at 15.74 / 14.41 tok/s, A at 14.14 / 14.72.
- **The answer length is not the cause either.** Past token 1,024 the rate drops to 12.1-13.7 tok/s, and a 2,233-token answer still averages 14.14 tok/s. That is not 8.4.
- The NVMe reads per decode token match G5's: ≈ 14.2 derived for G5, 12.0-14.6 here. So the same I/O cost about 70 ms/token here and 119 ms/token in G5.
- The ids are not reproducible run to run (see #215's acceptance comment). The four answers had 1,192-2,233 tokens on the same seed.
- **Remaining candidates:** Crow running beside serve (CPU-lane contention under `CROW_GLM_CPU_LANE=split`, untested by robin's rule), and the host draw. #215 measured S − D at 2.3 ms/token on this prompt, which is ≤ ~0.5 tok/s here. That share does not explain 8.4 vs 14.
