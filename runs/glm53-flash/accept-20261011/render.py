"""Offline render + tokenize of a chat body with GLM-5.3-Flash's own template and tokenizer
(jinja2 trim_blocks/lstrip_blocks/loopcontrols, tokenizers), the way #219's offline finding did.
Selftest: `python -I render.py` checks the prompt lengths of the G5 rows (6438, 8781, 8867, 6456, 14066).
"""
import json
import os
import sys

import jinja2
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bodies  # noqa: E402

TPL = open(os.path.join(bodies.MODEL_DIR, "chat_template.jinja"), encoding="utf-8").read()
TOK = Tokenizer.from_file(os.path.join(bodies.MODEL_DIR, "tokenizer.json"))


def _tojson(v, ensure_ascii=True, indent=None):
    return json.dumps(v, ensure_ascii=ensure_ascii, indent=indent)


ENV = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
ENV.filters["tojson"] = _tojson
TEMPLATE = ENV.from_string(TPL)


def norm(messages):
    out = []
    for m in messages:
        m = dict(m)
        if m.get("tool_calls"):
            tcs = []
            for tc in m["tool_calls"]:
                tc = json.loads(json.dumps(tc))
                a = tc["function"].get("arguments")
                if isinstance(a, str):
                    tc["function"]["arguments"] = json.loads(a)
                tcs.append(tc)
            m["tool_calls"] = tcs
        out.append(m)
    return out


def text(body, gen=True):
    return TEMPLATE.render(messages=norm(body["messages"]), tools=body.get("tools"), add_generation_prompt=gen)


def ids(body, gen=True):
    return TOK.encode(text(body, gen), add_special_tokens=False).ids


def tid(tok):
    return TOK.token_to_id(tok)


if __name__ == "__main__":
    for name, b, want in [("t1r1", bodies.t1(1), 6438), ("t1r2", bodies.t1(2), 8781), ("t1r3", bodies.t1(3), 8867),
                          ("t2r1", bodies.t2(1), 6456), ("t2r2", bodies.t2(2), 14066)]:
        print(name, len(ids(b)), "G5", want)
    a, b = ids(bodies.t1(1)), ids(bodies.t2(1))
    L = next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)
    c = ids(bodies.third())
    L3 = next(i for i, (x, y) in enumerate(zip(b, c)) if x != y)
    print("L(t1,t2)", L, "first <|user|> t1", a.index(tid("<|user|>")), "t2", b.index(tid("<|user|>")),
          "third len", len(c), "L(t2,third)", L3, "L(t1,third)", next(i for i, (x, y) in enumerate(zip(a, c)) if x != y))
    print(repr(text(bodies.t1(1))[-80:]))
