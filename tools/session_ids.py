"""Render a Crow session file into the token ids `serve` prefilled for it --
the calibration corpus for a hot set cut on real Crow traffic.

Usage (oracle venv, from the repository root):
  .venv-oracle/bin/python tools/session_ids.py <session.json> <out-ids.json> [<out-mask.json>]

With a third path it also writes the GENERATED positions: for every assistant
message, the span after the prompt the model was given (the render of the
messages before it plus the generation prompt) up to the end of its own
render -- its reasoning, text and tool calls. Those are the tokens decode
produced, so their routing is decode's routing (causal model). A span whose
prompt is not a token prefix of the longer render is skipped and counted.

The messages go through the model's own chat template (the one the container
carries, models/Qwen3.8-Flash-Next-original/chat_template.jinja) with their
reasoning and tool calls, thinking on -- the shape of a goal-mode request.
Tool schemas are not rendered: Crow's request carries them, the session file
does not, and they are a fixed prefix of a few thousand tokens.
"""
import json
import sys

from transformers import AutoTokenizer

MODEL = "models/Qwen3.8-Flash-Next-original"


def messages(path: str) -> list:
    doc = json.load(open(path, encoding="utf-8"))
    out = []
    for m in doc["messages"]:
        msg = {"role": m["role"], "content": m.get("content") or ""}
        if m.get("reasoning_content"):
            msg["reasoning_content"] = m["reasoning_content"]
        calls = m.get("tool_calls")
        if isinstance(calls, str):
            calls = json.loads(calls)
        if calls:
            msg["tool_calls"] = []
            for c in calls:
                fn = dict(c.get("function") or c)
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except ValueError:
                        args = {"raw": args}
                msg["tool_calls"].append({"type": "function", "function": {
                    "name": fn.get("name", ""), "arguments": args}})
        if m.get("tool_call_id"):
            msg["tool_call_id"] = m["tool_call_id"]
        out.append(msg)
    return out


def render(tok, msgs, gen=False) -> list:
    ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=gen)
    if hasattr(ids, "keys") and "input_ids" in ids:
        ids = ids["input_ids"]
    return [int(i) for i in ids]


def main() -> None:
    src, dst = sys.argv[1], sys.argv[2]
    tok = AutoTokenizer.from_pretrained(MODEL)
    msgs = messages(src)
    ids = render(tok, msgs)
    json.dump(ids, open(dst, "w"))
    print("%s: %d tokens -> %s" % (src, len(ids), dst))
    if len(sys.argv) < 4:
        return
    spans, skipped = [], 0
    for i, m in enumerate(msgs):
        if m["role"] != "assistant" or i == 0:
            continue
        prompt = render(tok, msgs[:i], gen=True)
        upto = render(tok, msgs[:i + 1])
        if upto[:len(prompt)] != prompt or ids[:len(upto)] != upto:
            skipped += 1
            continue
        spans.append([len(prompt), len(upto)])
    json.dump({"spans": spans, "skipped": skipped, "tokens": len(ids)},
              open(sys.argv[3], "w"))
    print("generated positions: %d in %d spans, %d assistant messages skipped"
          % (sum(b - a for a, b in spans), len(spans), skipped))


if __name__ == "__main__":
    main()
