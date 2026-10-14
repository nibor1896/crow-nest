"""Replays for #219, #217, #216 and the combined run (2026-10-11 block). One fresh serve boot per replay,
arm4.env + the arm's switch, local main 2a18fbc, port 8099, nothing else on the GPU.

  219       6 replays off,on,off,on,off,on (CROW_GLM_ANSWER_REUSE): request 1 = t2 request 1 (Crow body, seed
            1145815877) with crow_force_ids = force219.json (7,547 ids: canonical answer, 'XL' at offset 6,669 as
            'X'+'L', then <|observation|>); request 2 = t2 messages 0-3, temperature 0, logprobs + top_logprobs 20 +
            crow_force_ids [] (crow_id per generated id), max_tokens 4096.
  217       6 replays off,on,... (CROW_GLM_PREFIX_ANCHOR): t1 r1, t2 r1, third chat r1 (amendment
            issuecomment-6109018083), max_tokens 64; the third with logprobs + top_logprobs 20.
  216step0  1 replay, switch off: t1 r1-r3, Crow bodies (sampled, logged seeds, max_tokens 16384).
  216       5 more replays on,off,on,off,on (CROW_GLM_SMALL_PROMPT=1).
  combined  2 replays: all off (switches off, logit_bias {"0": 0} = the host draw) then all on (ANSWER_REUSE,
            PREFIX_ANCHOR, SMALL_PROMPT = 1, device draw); the 9 G5 requests t1 r1-r3, t2 r1-r6, Crow bodies with
            the logged seeds, each answer forced to the stored one (chain.py) so every prompt extends the held ids.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bodies  # noqa: E402
import harness as h  # noqa: E402

ON = {"219": {"CROW_GLM_ANSWER_REUSE": "1"}, "217": {"CROW_GLM_PREFIX_ANCHOR": "1"},
      "216": {"CROW_GLM_SMALL_PROMPT": "1"}}
ALL_ON = {"CROW_GLM_ANSWER_REUSE": "1", "CROW_GLM_PREFIX_ANCHOR": "1", "CROW_GLM_SMALL_PROMPT": "1"}


def replay(ticket, tag, switches, reqs):
    out = os.path.join(h.HERE, ticket)
    os.makedirs(out, exist_ok=True)
    rows = os.path.join(out, "rows.jsonl")
    with h.Serve(os.path.join(out, f"serve-{tag}.log"), switches) as sv:
        h.jdump(rows, {"event": "boot", "tag": tag, "switches": switches, "boot_s": sv.boot_s, "boot_err": sv.boot_err,
                       "vram_before": sv.vram_before, "perf": sv.perf_before, "boot_lines": sv.boot_lines()})
        print(ticket, tag, "boot", sv.boot_s, sv.boot_err, flush=True)
        if sv.boot_err:
            return
        prev = None
        for name, body in reqs:
            res = sv.chat(body)
            c = h.counters(res)
            s = h.summary(res)
            row = {"event": "req", "tag": tag, "req": name, **s, "counters": c,
                   "nvme_delta": (c[2] - prev[2]) if c and prev else (c[2] if c else None),
                   "ids": res.get("ids"), "first_logprobs": res.get("first_logprobs"),
                   "reasoning": res.get("reasoning"), "content": res.get("content"),
                   "tool_calls": res.get("tool_calls"), "usage": res.get("usage"), "log": res.get("log"),
                   "routing": res.get("routing"), "wall_s": res.get("wall_s")}
            prev = c or prev
            h.jdump(rows, row)
            print(ticket, tag, name, s, flush=True)
            if not res.get("serve_alive"):
                h.jdump(rows, {"event": "crash", "tag": tag, "req": name})
                return
    h.jdump(rows, {"event": "stop", "tag": tag, "rc": sv.alive_at_exit})


def reqs_219():
    forced = json.load(open(os.path.join(h.HERE, "force219.json")))["forced"]
    r1 = bodies.t2(1, crow_force_ids=forced)
    r2 = bodies.t2(2, max_tokens=4096, temperature=0, logprobs=True, top_logprobs=20, crow_force_ids=[])
    return [("r1", r1), ("r2", r2)]


def reqs_217():
    return [("t1r1", bodies.t1(1, max_tokens=64)), ("t2r1", bodies.t2(1, max_tokens=64)),
            ("third", bodies.third(max_tokens=64, logprobs=True, top_logprobs=20))]


def reqs_216():
    return [("t1r1", bodies.t1(1)), ("t1r2", bodies.t1(2)), ("t1r3", bodies.t1(3))]


def reqs_combined(host):
    import chain
    out = []
    for r in chain.chain():
        b = dict(r["body"], crow_force_ids=r["forced"])
        if host:
            b["logit_bias"] = {"0": 0}
        out.append((r["name"], b))
    return out


def alternate(ticket, n, start_on):
    for k in range(n):
        on = (k % 2 == 0) == start_on
        idx = k + 1
        yield (f"{'on' if on else 'off'}-r{idx}", ON[ticket] if on else {})


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "219":
        for tag, sw in alternate("219", 6, False):
            replay("219", tag, sw, reqs_219())
    elif what == "217":
        for tag, sw in alternate("217", 6, False):
            replay("217", tag, sw, reqs_217())
    elif what == "216step0":
        replay("216", "off-r1", {}, reqs_216())
    elif what == "216":
        for k, on in enumerate([True, False, True, False, True]):
            replay("216", f"{'on' if on else 'off'}-r{k + 2}", ON["216"] if on else {}, reqs_216())
    elif what == "combined":
        replay("combined", "all-off", {}, reqs_combined(True))
        replay("combined", "all-on", ALL_ON, reqs_combined(False))
    elif what == "selftest":
        for f in (reqs_219, reqs_217, reqs_216):
            print(f.__name__, [(n, len(json.dumps(b)), {k: b[k] for k in b if k not in ("messages", "tools", "crow_force_ids")})
                               for n, b in f()])
        print("combined", [(n, len(b["crow_force_ids"]), "logit_bias" in b) for n, b in reqs_combined(True)])
        print(list(alternate("219", 6, False)))
