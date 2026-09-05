"""Tokenize text with the reference tokenizer (transformers 5.16.1, oracle
venv) — the bridge between text prompts and the engine's id input.
Usage:
  python tools/tokenize_ids.py "text"        → raw ids (no special tokens)
  echo "text" | python tools/tokenize_ids.py --chat
                                             → chat-template ids (same token
                                               stream the llama.cpp arm gets:
                                               --jinja + enable_thinking:false).
Long text via stdin: the Windows command line caps at ~32k chars (measured
2026-09-03 with the 60k t1-read prompt)."""
import json
import sys

from transformers import AutoTokenizer

MODEL = "models/Qwen3.8-Flash-Next-original"

tok = AutoTokenizer.from_pretrained(MODEL)
args = list(sys.argv[1:])
chat = "--chat" in args
if chat:
    args.remove("--chat")
# prompt text via stdin when no argv: long prompts overflow the Windows
# command line (~32k chars)
text = args[0] if args else sys.stdin.read()
if chat:
    # parity harness fairness (fable gate, 2026-09-03): the crow arm must
    # receive the SAME token stream as the llama.cpp arm — chat template
    # with thinking disabled, generation prompt appended
    out = tok.apply_chat_template(
        [{"role": "user", "content": text}],
        add_generation_prompt=True,
        tokenize=True,
        enable_thinking=False,
    )
    # transformers 5.x returns a BatchEncoding (not a dict subclass):
    # out["input_ids"] is already the flat id list for this single message
    if hasattr(out, "keys") and "input_ids" in out:
        ids = list(out["input_ids"])
    elif out and isinstance(out[0], (list, tuple)):
        ids = list(out[0])
    else:
        ids = list(out)
else:
    ids = tok(text, add_special_tokens=False)["input_ids"]
print(json.dumps(list(ids)))
