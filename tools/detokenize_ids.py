"""Detokenize greedy answer ids from the parity harness (batch).
Usage: python tools/detokenize_ids.py <in.json>   where in.json = {"key": {"answer_ids": [...]}}
Prints {"key": "text"}."""
import json
import sys

from transformers import AutoTokenizer

MODEL = "models/Qwen3.8-Flash-Next-original"

tok = AutoTokenizer.from_pretrained(MODEL)
src = json.load(open(sys.argv[1], encoding="utf-8"))
out = {}
for key, val in src.items():
    ids = val["answer_ids"] if isinstance(val, dict) else val
    out[key] = tok.decode([int(i) for i in ids], skip_special_tokens=True)
print(json.dumps(out, ensure_ascii=False))
