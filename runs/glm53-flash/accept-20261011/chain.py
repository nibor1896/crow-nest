"""The G5 t1 + t2 request chain for the combined run: per request the Crow body (stored messages, the logged seed) and
the stored answer as forced ids, so every next prompt extends what the engine generated, as in G5.
Forced ids = the answer's canonical ids from the offline render up to the stop (<|observation|> after a tool call,
<|user|> after a final answer), with G5's two measured non-canonical splits put back (#219 Evidence 3):
t1 answer 1 offset 574 ' fishermen' -> ' fish' + 'ermen'; t2 answer 1 offset 6,669 'XL' -> 'X' + 'L'.
Selftest: `python -I chain.py` prints prompt lengths and forced lengths against G5."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bodies  # noqa: E402
import render  # noqa: E402

OBS, USER = render.tid("<|observation|>"), render.tid("<|user|>")


def seeds(path):
    out = []
    for t in json.load(open(path, encoding="utf-8"))["timings"]:
        out += t.get("seeds") or []
    return out


def answer_ids(msgs, n):
    """ids of message n (assistant) after the prompt msgs[:n], up to and including the stop id."""
    p = render.ids({"messages": msgs[:n], "tools": bodies.tools()})
    nxt = n + 1 < len(msgs)
    full = render.ids({"messages": msgs[:n + 2] if nxt else msgs[:n + 1], "tools": bodies.tools()}, gen=False)
    assert full[:len(p)] == p
    if nxt:
        end = full.index(OBS, len(p))
        return p, full[len(p):end] + [OBS]
    tail = full[len(p):]
    return p, tail + [USER]


def split(ids, off, want_tok, parts):
    assert render.TOK.id_to_token(ids[off]) == want_tok, (render.TOK.id_to_token(ids[off]), want_tok)
    new = [render.tid(t) for t in parts]
    assert None not in new and render.TOK.decode(new) == render.TOK.decode([ids[off]])
    return ids[:off] + new + ids[off + 1:]


def chain():
    t1m, t2m = bodies.msgs(bodies.T1_FILE), bodies.msgs(bodies.T2_FILE)
    s1, s2 = seeds(bodies.T1_FILE), seeds(bodies.T2_FILE)
    reqs = []
    for k, n in enumerate((2, 5, 7)):
        p, f = answer_ids(t1m, n)
        if k == 0:
            f = split(f, 574, "Ġfishermen", ["Ġfish", "ermen"])
        reqs.append({"name": f"t1r{k + 1}", "body": bodies.body(t1m[:n], s1[k]), "prompt_len": len(p), "forced": f})
    for k, n in enumerate((2, 4, 8, 12, 16, 18)):
        p, f = answer_ids(t2m, n)
        if k == 0:
            f = split(f, 6669, "XL", ["X", "L"])
        reqs.append({"name": f"t2r{k + 1}", "body": bodies.body(t2m[:n], s2[k]), "prompt_len": len(p), "forced": f})
    return reqs


if __name__ == "__main__":
    for r in chain():
        print(r["name"], "prompt", r["prompt_len"], "forced", len(r["forced"]), "seed", r["body"]["seed"],
              "body B", len(json.dumps(r["body"])))
