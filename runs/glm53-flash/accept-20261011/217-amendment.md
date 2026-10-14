## Amendment to the Expected result (2026-10-11, written before any #217 after-row exists)

**Change:** the replay gets a **third new chat**, and A1 / A2 are judged on its first request instead of t2's.

**Reason:** Crow's stored G5 chats carry system prompts that differ from char 363 on. That char is the working-area line: `...\g5-run1\t1-wordfreq` in t1 vs `...\g5-run1\t2-roman` in t2. The tools block and the system message come before the first `<|user|>`. So under this ticket's own placement rule, t2's first request is COLD and sets the anchor at its L = 6,043. Only a third chat can start WARM from that anchor. The implementer found this offline (implementation comment above). The offline render confirms it: GLM's `chat_template.jinja` + `tokenizer.json`, jinja2 3.1.6, tokenizers 0.22.2. It reproduces the G5 prompt lengths 6,438 / 8,781 / 8,867 / 6,456 / 14,066 exactly:
- t1 request 1: 6,438 ids; t2 request 1: 6,456 ids.
- The first `<|user|>` is at position 6,376 in both.
- L(t1, t2) = 6,043.

**Third chat:**
- t2's system message, with the working area replaced by `C:\Users\robin\Desktop\g5-run1\t3-accept`, and t1's user message.
- Crow's 31 tools and Crow's sampling fields (temperature 1, top_p 0.95, min_p 0.01, max_tokens as below), seed 1408639702.
- Offline: 6,438 ids, and L against t2 request 1 is 6,043.

**Replay per boot:** a fresh `serve` with `runs/glm53-flash/quick/arm4.env`, then `CROW_GLM_PREFIX_ANCHOR=0` vs `=1`, the same binary (local main `2a18fbc`), 3 replays per arm, alternated, nothing else on the GPU.
- Requests in order: t1 request 1 (seed 1408639702), t2 request 1 (seed 1145815877), third chat request 1.
- The bodies are rebuilt byte for byte as Crow 3.2.9 sends them. The t1 bodies match the G5 log's body sizes: 27,159 / 36,479 / 36,867 B.
- `max_tokens` is 64 on all three. This criterion measures the prompt phase, and the answer length moves neither L nor the anchor.

**Criteria (thresholds unchanged):**
- **A1:** the third chat's request 1 in the after arm is WARM with prefilled ≤ (prompt_len − L) + 64, on 3 of 3 replays. Its `[cache]` line names the anchor. Expected after: L 6,043, ≤ 459.
- **A2:** that request's median `prefill_ms` is lower after than before. The saving per new chat is reported in s.
- **A3:** unchanged. t1 request 1's median `prefill_ms` after ≤ 1.03 × before.
- **A4:**
  - The two GPU tests, unchanged.
  - The replay part, on the third chat's request 1: a test build that dumps the snapshot's logits row is a code change and is out of scope for this block. So the prompt-end row is read from serve's own `logprobs: true, top_logprobs: 20` at the first generated position. That is the raw distribution (log-softmax of the lm_head row) at the prompt's end. The block reports cosine over those 20 entries (aligned by id) and top-1 equality between the arms. This is a partial view of the row and is reported as such.
- **A5:** unchanged.
- t2 request 1 is reported in both arms. Expected: COLD in both; in the after arm, an anchor at 6,043.
