"""Build the harness-format ten-task series: 6 hash-pinned seeds from
docs/ten-task-prompts-crowlab.json + the 4 authored t7-t10 prompts
(decode_out/ten-tasks-authored-t7-t10.json; t7 embeds the full
engine/src/kernels.rs as its long-context material).
Output: decode_out/ten-tasks.json  [{"id","text","max_tokens"}, ...]
Re-run after editing the authored file or the kernels source."""
import json

SEED_MAX_TOKENS = {
    # 2026-09-03 rev2 (protocol finding W2-W6): 640-896 clipped 7/10 llama
    # reference answers before completion — full passes were impossible for
    # BOTH arms. Budgets must carry the complete expected answer; the model
    # runs with enable_thinking:false (no thinking tokens to budget for).
    "t1-read": 1024,  # summary + answer over a long C++ source
    "t2-write": 1280, # header + unit tests (code is token-heavy)
    "t3-debug": 1024, # root cause + explanation
    "t4-prose": 1024, # technical prose
    "t5-agent": 1536, # plan + steps
    "t6-reason": 1536,# derivation + worked case
}

seeds = json.load(open("docs/ten-task-prompts-crowlab.json", encoding="utf-8"))
authored = json.load(open("decode_out/ten-tasks-authored-t7-t10.json", encoding="utf-8"))["tasks"]
kernels = open("engine/src/kernels.rs", encoding="utf-8").read()

series = []
for tid, text in seeds.items():
    series.append({"id": tid, "text": text, "max_tokens": SEED_MAX_TOKENS[tid]})
for t in authored:
    text = t["text"] if "text" in t else t["text_prefix"] + kernels + t["text_suffix"]
    series.append({"id": t["id"], "text": text, "max_tokens": t["max_tokens"]})

ids = [s["id"] for s in series]
assert len(ids) == 10 and len(set(ids)) == 10, ids
json.dump(series, open("decode_out/ten-tasks.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("written decode_out/ten-tasks.json:", ", ".join(ids))
