#!/usr/bin/env python3
"""#91 MULTI-SITE teacher-forced corruption probe: one number per arm in minutes.

WHY: the K=2 replay (tools/corruption-replay-probe.py + teacher-forced-compare.py, MEAS-0923)
measures ONE token (#65, `/home/nibor1` -> `8` vs `1`). One site cannot rank fixes. This tool
holds ~20 corrupted sites of the 2026-09-23 diorama session (o[13]->o[113], N3->N33,
mat4->mat44, texImage3D->texImage33D, 128->1 28, /home/nibor1896->nibor1196, markup leaks, ...)
and, per arm, teacher-forces the model's own produced tokens up to each site and reads
lp(correct) and lp(corrupt) there.

THE REQUEST, per site (session segment S, assistant message K, tool call C):
  - messages   the segment file's messages[0:K]: Crow's save writes conversation.payload(), so
               the file is the wire history. The head (messages[0]) of a segment file is the head
               at SAVE time; within a segment it held for every request checked (`build` matches
               the rendered prompt token count against serve's `[chat] prompt N tok` line of the
               live request, engine.log, and records the delta per site: `live_prompt_tokens`).
  - the body   built BY CROW (crow_core.stream_reply with a capturing `_post_stream`, the
               snapshot c4f1b3c the GUI ran), then: `reasoning_budget_tokens` /
               `reasoning_budget_message` removed (serve refuses them together with
               `crow_force_ids`; they act on the sampler only, the prompt bytes are unchanged),
               `temperature 0`, `tool_choice "none"` (no grammar; the template still renders the
               tools), `stream false`, `logprobs true`, `top_logprobs 20`, `crow_force_ids`.
  - the forced ids  the tokens of the assistant turn as the chat template renders it
               (`<think>\\n` is the end of the generation prompt; then reasoning|trim,
               `\\n</think>\\n\\n`, content, the `<tool_call>` markup with the arguments exactly as
               stored), up to and INCLUDING the corrupt token, tokenized with the model's own
               tokenizer (.venv-oracle, `build` only). The re-tokenization of text is the one
               approximation: the live ids are not logged (engine.log carries no chat=debug ids for
               robin's 04:37Z boot). `build` records how many tokens the rendered turn has against
               serve's `generated N tok` for the live request.

AT THE SITE (entry j = the corrupt token as produced, forced):
  lp_corrupt = the entry's own logprob (exact: it is the forced id),
  lp_correct = the logprob of the correct id among the top-20 alternatives (by crow_id); when it
               is not among them, a second request forces ids[:j] + [correct] and reads it exactly
               (`correct_forced: true`),
  top1       = the first alternative,
  margin     = lp_correct - lp_corrupt (positive = the arm prefers the correct token),
  corrupt_wins = lp_corrupt > lp_correct.
Arm metric: corrupt-win fraction over the sites, mean margin, and the per-site table.

Subcommands:
  build  (needs .venv-oracle: transformers)  sites spec -> sites JSON with forced ids
         .venv-oracle/bin/python tools/multisite-corruption-probe.py build \\
             --spec tools/corpora/91-multisite-0923-spec.json --out tools/corpora/91-multisite-0923.json
  run    (plain python3)  one arm against a running serve
         python3 tools/multisite-corruption-probe.py run --sites tools/corpora/91-multisite-0923.json \\
             --port 8111 --label bare --json OUT.json
  table  compare arm JSONs
         python3 tools/multisite-corruption-probe.py table A.json B.json ... [--md OUT.md]
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
TOP = 20


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def resolve(path, base):
    path = os.path.expanduser(path)
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(base, path))


# ------------------------------------------------------------------ crow's request

class _Captured(Exception):
    pass


def load_crow(path):
    spec = importlib.util.spec_from_file_location("crow_core_multisite", path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(path))
    spec.loader.exec_module(mod)
    return mod


def crow_body(crow, messages, served_model, wire_model="crow", base_url="http://127.0.0.1:8099/v1"):
    """The body crow_core.stream_reply builds for this history, captured, never sent.
    `served_model` is what /props reported (the session's `model`), `wire_model` the window's
    --model default; the GUI passes both (crow_gui.py, #220)."""
    got = {}

    def capture(url, body, api_key, timeout, extra=None):
        got["url"], got["body"] = url, body
        raise _Captured()

    saved = crow._post_stream
    crow._post_stream = capture
    try:
        conv = crow.Conversation()
        conv._messages = copy.deepcopy(messages)
        s = crow.sampling_for(served_model)
        try:
            crow.stream_reply(conv, base_url=base_url, model=wire_model, api_key="",
                              temperature=s["temperature"], top_p=s["top_p"], min_p=s["min_p"],
                              top_k=s.get("top_k"), presence_penalty=s.get("presence_penalty"),
                              served_name=served_model, timeout=10)
        except _Captured:
            pass
    finally:
        crow._post_stream = saved
    if "body" not in got:
        sys.exit("crow_core.stream_reply returned without reaching _post_stream")
    return got["url"], got["body"]


def forced_body(body, force_ids):
    """The live body turned into a teacher-forced measurement request (see the docstring)."""
    b = dict(body)
    for k in ("reasoning_budget_tokens", "reasoning_budget_message", "stream_options", "seed"):
        b.pop(k, None)
    b.update({"temperature": 0, "tool_choice": "none", "stream": False, "logprobs": True,
              "top_logprobs": TOP, "crow_force_ids": list(force_ids),
              "max_tokens": len(force_ids) + 1})
    return b


# ------------------------------------------------------------------ build (tokenizer side)

def normalize_for_template(messages):
    """serve's normalize_messages: a JSON-string `arguments` becomes the object it encodes."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("tool_calls"):
            tcs = []
            for tc in m["tool_calls"]:
                tc = copy.deepcopy(tc)
                a = (tc.get("function") or {}).get("arguments")
                if isinstance(a, str) and a != "":
                    try:
                        a = json.loads(a)
                    except json.JSONDecodeError:
                        a = {"_raw": a}
                    if not isinstance(a, dict):
                        a = {"_raw": json.dumps(a, separators=(",", ":"))}
                elif a is None:
                    a = {}
                tc["function"]["arguments"] = a
                tcs.append(tc)
            m["tool_calls"] = tcs
        out.append(m)
    return out


def locate_site(cont, call_start, param, produced, correct, occurrence=0):
    """byte-free CHAR index (in `cont`) of the first char where `produced` and `correct` differ,
    searched inside tool call text from `call_start`, within parameter `param`."""
    ptag = "<parameter=%s>\n" % param
    p0 = cont.find(ptag, call_start)
    if p0 < 0:
        raise ValueError("parameter %r not found in the rendered call" % param)
    p1 = cont.find("\n</parameter>", p0 + len(ptag))
    at, start = -1, p0 + len(ptag)
    for _ in range(occurrence + 1):
        at = cont.find(produced, start)
        if at < 0 or (p1 >= 0 and at > p1 and not produced.endswith("</parameter>")):
            raise ValueError("produced %r (occurrence %d) not in parameter %r" % (produced, occurrence, param))
        start = at + 1
    cp = 0
    while cp < min(len(produced), len(correct)) and produced[cp] == correct[cp]:
        cp += 1
    if cp == len(produced):
        raise ValueError("produced %r is a prefix of correct %r" % (produced, correct))
    return at, at + cp, cp


def build(args):
    from transformers import AutoTokenizer                     # .venv-oracle only
    spec_path = os.path.abspath(args.spec)
    with open(spec_path) as fh:
        spec = json.load(fh)
    base = os.path.dirname(spec_path)
    tok = AutoTokenizer.from_pretrained(resolve(spec["tokenizer"], base))
    crow_core = resolve(spec["crow_core"], base)
    crow = load_crow(crow_core)
    sessions = {}
    for seg, s in spec["segments"].items():
        path = resolve(s["file"], base)
        with open(path, "rb") as fh:
            blob = fh.read()
        sessions[seg] = {"doc": json.loads(blob), "sha256": hashlib.sha256(blob).hexdigest(),
                         "file": s["file"], "req_range": s.get("engine_requests")}
    live = parse_engine_log(resolve(spec["engine_log"], base), spec.get("engine_log_after")) \
        if spec.get("engine_log") else []
    served = spec["served_model"]
    out_sites, rendered_cache = [], {}
    for i, st in enumerate(spec["sites"]):
        seg, K, C = st["segment"], st["at"], st.get("call", 0)
        sess = sessions[seg]
        msgs = sess["doc"]["messages"]
        if msgs[K]["role"] != "assistant":
            raise SystemExit("site %s: messages[%d] is not an assistant turn" % (st["id"], K))
        key = (seg, K)
        if key not in rendered_cache:
            _, body = crow_body(crow, msgs[:K], served)
            tools = body.get("tools")
            if not tools or body.get("reasoning_effort") != "high":
                raise SystemExit("crow_core built a body without tools / reasoning_effort high: "
                                 "copy manifests/ beside its cli/ directory")
            effort = spec.get("template_reasoning_effort", "xhigh")
            prompt = tok.apply_chat_template(normalize_for_template(msgs[:K]), tools=tools, tokenize=False,
                                              add_generation_prompt=True, reasoning_effort=effort)
            full = tok.apply_chat_template(normalize_for_template(msgs[:K + 1]), tools=tools, tokenize=False,
                                            add_generation_prompt=False, reasoning_effort=effort)
            if not full.startswith(prompt):
                raise SystemExit("site %s: the rendered history is not a prefix of the rendered turn" % st["id"])
            cont = full[len(prompt):]
            if cont.endswith("<|im_end|>\n"):
                cont = cont[:-len("<|im_end|>\n")]
            n_prompt = len(tok.encode(prompt, add_special_tokens=False))
            enc = tok(cont, add_special_tokens=False, return_offsets_mapping=True)
            match = [r for r in live if r["prompt"] == n_prompt and
                     (not sess["req_range"] or sess["req_range"][0] <= r["n"] <= sess["req_range"][1])]
            rendered_cache[key] = {"cont": cont, "ids": enc["input_ids"], "offs": enc["offset_mapping"],
                                   "n_prompt": n_prompt, "live": match[0] if match else None,
                                   "body_bytes": len(json.dumps(body).encode("utf-8"))}
        r = rendered_cache[key]
        if st.get("live_request") and not r["live"]:
            # an image in the history: serve renders its patch tokens, the offline template one
            # pad token, so the counts cannot match; the spec names the live request (by its
            # generated count) and `run` checks serve's prompt_tokens against it
            r["live"] = next(x for x in live if x["n"] == st["live_request"])
            r["live_by"] = "spec (history holds images)"
        cont, ids, offs = r["cont"], r["ids"], r["offs"]
        # the call's start in the rendered turn
        cs = -1
        for _ in range(C + 1):
            cs = cont.find("<tool_call>", cs + 1)
        if cs < 0:
            raise SystemExit("site %s: tool call %d not in the rendered turn" % (st["id"], C))
        at, div, cp = locate_site(cont, cs, st["param"], st["produced"], st["correct"], st.get("occurrence", 0))
        j = next(t for t, (a, b) in enumerate(offs) if a <= div < b)
        # the correct continuation re-tokenized: same text up to `div`, then the correct rest and
        # the produced text after the produced span (context for the tokenizer's merges only)
        alt_text = cont[:at] + st["correct"] + cont[at + len(st["produced"]):]
        alt_ids = tok(alt_text, add_special_tokens=False)["input_ids"]
        if alt_ids[:j] != ids[:j]:
            raise SystemExit("site %s: the correct text tokenizes differently before token %d" % (st["id"], j))
        corrupt_id, correct_id = ids[j], alt_ids[j]
        if corrupt_id == correct_id:
            raise SystemExit("site %s: correct and corrupt give the same token %d" % (st["id"], j))
        out_sites.append({
            "id": st["id"], "segment": seg, "at": K, "call": C, "param": st["param"],
            "class": st.get("class", ""), "note": st.get("note", ""), "context": st.get("context", ""),
            "produced": st["produced"], "correct": st["correct"],
            "token_index": j, "corrupt_id": corrupt_id, "correct_id": correct_id,
            "corrupt_token": tok.decode([corrupt_id]), "correct_token": tok.decode([correct_id]),
            "context_before": cont[max(0, div - 60):div],
            "rendered_prompt_tokens": r["n_prompt"],
            "live_request": r["live"]["n"] if r["live"] else None,
            "live_prompt_tokens": r["live"]["prompt"] if r["live"] else None,
            "live_generated": r["live"]["gen"] if r["live"] else None,
            "live_match": r.get("live_by", "rendered prompt tokens == serve's prompt tokens") if r["live"] else None,
            "rendered_turn_tokens": len(ids) + 1,          # + <|im_end|>
        })
        print("%-14s seg %s K %3d j %5d prompt %6d live #%s %s gen %s/%d  %r -> corrupt %r correct %r" % (
            st["id"], seg, K, j, r["n_prompt"], r["live"]["n"] if r["live"] else "-",
            "=" if r["live"] else "NO MATCH", r["live"]["gen"] if r["live"] else "-", len(ids) + 1,
            cont[max(0, div - 25):div], tok.decode([corrupt_id]), tok.decode([correct_id])), file=sys.stderr)
    # one forced sequence per (segment, K): the longest prefix any of its sites needs
    calls = []
    for key in sorted(rendered_cache, key=lambda k: (k[0], k[1])):
        mine = [s for s in out_sites if (s["segment"], s["at"]) == key]
        jmax = max(s["token_index"] for s in mine)
        calls.append({"segment": key[0], "at": key[1], "force_ids": rendered_cache[key]["ids"][:jmax + 1],
                      "sites": [s["id"] for s in mine]})
    # free-running greedy points: the turn forced up to the first byte of a parameter value,
    # then the arm writes on its own (not teacher-forced) and `freerun` counts digit errors
    freeruns = []
    for fr in spec.get("freerun", []):
        seg, K = fr["segment"], fr["at"]
        msgs = sessions[seg]["doc"]["messages"]
        _, body = crow_body(crow, msgs[:K], served)
        effort = spec.get("template_reasoning_effort", "xhigh")
        prompt = tok.apply_chat_template(normalize_for_template(msgs[:K]), tools=body.get("tools"), tokenize=False,
                                          add_generation_prompt=True, reasoning_effort=effort)
        full = tok.apply_chat_template(normalize_for_template(msgs[:K + 1]), tools=body.get("tools"), tokenize=False,
                                        add_generation_prompt=False, reasoning_effort=effort)
        cont = full[len(prompt):]
        ptag = "<parameter=%s>\n" % fr["param"]
        cut = cont.find(ptag, cont.find("<tool_call>")) + len(ptag)
        head = cont[:cut] + fr.get("value_prefix", "")
        ids = tok(head, add_special_tokens=False)["input_ids"]
        freeruns.append(dict(fr, force_ids=ids, forced_text_tail=head[-120:]))
        print("freerun %-12s seg %s K %d: %d forced tokens, then %d free" % (fr["id"], seg, K, len(ids),
                                                                            fr.get("max_new", 600)), file=sys.stderr)
    doc = {"what": "#91 multi-site teacher-forced corruption set, 2026-09-23 diorama session",
           "built_by": "tools/multisite-corruption-probe.py build", "spec": os.path.relpath(spec_path, REPO),
           "served_model": served, "crow_core": spec["crow_core"], "crow_core_sha256": sha256_file(crow_core),
           "segments": {k: {"file": v["file"], "sha256": v["sha256"]} for k, v in sessions.items()},
           "top_logprobs": TOP, "sites": out_sites, "requests": calls, "freerun": freeruns}
    with open(args.out, "w") as fh:
        json.dump(doc, fh, indent=1)
        fh.write("\n")
    nf = sum(len(c["force_ids"]) for c in calls)
    print("%d sites in %d requests, %d forced tokens in total -> %s" % (len(out_sites), len(calls), nf, args.out),
          file=sys.stderr)


def parse_engine_log(path, after=None):
    """serve's per-request lines: [{n, prompt, cached, gen, body}] in order (after the `after`
    timestamp prefix, e.g. the boot of record)."""
    import re
    out, cur, n = [], {}, 0
    with open(path, errors="replace") as fh:
        for line in fh:
            if after and line[:len(after)] < after:
                continue
            m = re.search(r"\[chat\] prompt (\d+) tok \((\d+) cached.*generated (\d+) tok", line)
            if m:
                cur = {"prompt": int(m[1]), "cached": int(m[2]), "gen": int(m[3])}
            m = re.search(r"POST /v1/chat/completions \(body (\d+) bytes\)", line)
            if m:
                n += 1
                out.append(dict(cur, n=n, body=int(m[1]), ts=line[:23]))
                cur = {}
    return out


# ------------------------------------------------------------------ run (serve side)

def post(url, body, timeout=7200):
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def entries_of(ans):
    ch = ans["choices"][0]
    return list(((ch.get("logprobs") or {}).get("content")) or [])


def read_site(entry, site):
    """lp_corrupt (exact, forced), lp_correct (from the alternatives or None), top1."""
    alts = entry.get("top_logprobs") or []
    if entry.get("crow_id") is not None and entry["crow_id"] != site["corrupt_id"]:
        raise ValueError("site %s: forced id %s is not the corrupt id %s" % (
            site["id"], entry.get("crow_id"), site["corrupt_id"]))
    lp_corrupt = entry["logprob"]
    lp_correct = next((a["logprob"] for a in alts if a.get("crow_id") == site["correct_id"]), None)
    top1 = alts[0] if alts else {}
    return {"lp_corrupt": lp_corrupt, "lp_correct": lp_correct,
            "top1_id": top1.get("crow_id"), "top1_token": top1.get("token"), "top1_lp": top1.get("logprob"),
            "min_top_lp": min((a["logprob"] for a in alts), default=None)}


def summarize(rows):
    ok = [r for r in rows if r.get("lp_correct") is not None]
    wins = sum(1 for r in ok if r["corrupt_wins"])
    margins = [r["margin"] for r in ok]
    return {"sites": len(rows), "sites_measured": len(ok), "corrupt_wins": wins,
            "corrupt_win_fraction": round(wins / len(ok), 4) if ok else None,
            "mean_margin": round(sum(margins) / len(margins), 4) if margins else None,
            "median_margin": round(sorted(margins)[len(margins) // 2], 4) if margins else None,
            # at Crow's temperature 1.0 the chance the corrupt token is SAMPLED at the site
            "mean_p_corrupt": round(sum(math.exp(r["lp_corrupt"]) for r in ok) / len(ok), 4) if ok else None,
            "correct_top1": sum(1 for r in ok if r["top1_id"] == r["correct_id"]),
            "corrupt_top1": sum(1 for r in ok if r["top1_id"] == r["corrupt_id"])}


def run(args):
    sites_path = os.path.abspath(args.sites)
    with open(sites_path) as fh:
        doc = json.load(fh)
    base = os.path.dirname(os.path.abspath(args.sites))
    spec_dir = os.path.dirname(resolve(doc["spec"], REPO))
    crow_core = resolve(args.crow_core or doc["crow_core"], spec_dir)
    if sha256_file(crow_core) != doc["crow_core_sha256"]:
        sys.exit("crow_core %s differs from the one the site set was built with" % crow_core)
    crow = load_crow(crow_core)
    sessions = {}
    for seg, s in doc["segments"].items():
        path = resolve(s["file"], spec_dir)
        with open(path, "rb") as fh:
            blob = fh.read()
        if hashlib.sha256(blob).hexdigest() != s["sha256"]:
            sys.exit("segment %s: %s is not the snapshot the site set was built on" % (seg, path))
        sessions[seg] = json.loads(blob)
    by_id = {s["id"]: s for s in doc["sites"]}
    only = set(args.only.split(",")) if args.only else None
    url = (args.base_url or "http://127.0.0.1:%d/v1" % args.port).rstrip("/") + "/chat/completions"
    rows, reqs, t_all = [], [], time.time()
    chained = {}                                   # segment -> last K whose prompt was sent
    for rq in doc["requests"]:
        sids = [s for s in rq["sites"] if not only or s in only]
        if not sids:
            continue
        if args.warm_chain:
            # WARM CHAIN: the live KV at a site was built request by request (engine.log: every
            # request after the segment's first is `[cache] WARM`, P = the previous prompt), so
            # the prompt of every earlier assistant turn of the segment is sent first, one
            # generated token each, and the site request rolls back to the previous prompt's
            # snapshot and prefills only its delta -- the prefill boundaries of the live session.
            seg_msgs = sessions[rq["segment"]]["messages"]
            first = [k for k in range(2, len(seg_msgs)) if seg_msgs[k]["role"] == "assistant"][1]
            t0, n_chain = time.time(), 0
            for k in range(max(first, chained.get(rq["segment"], 0) + 1), rq["at"]):
                if seg_msgs[k]["role"] != "assistant":
                    continue
                _, cb = crow_body(crow, seg_msgs[:k], doc["served_model"])
                post(url, forced_body(cb, []))
                n_chain += 1
            chained[rq["segment"]] = rq["at"]
            print("  warm chain seg %s up to K %d: %d requests, %.1f s" % (
                rq["segment"], rq["at"], n_chain, time.time() - t0), file=sys.stderr)
        msgs = sessions[rq["segment"]]["messages"][:rq["at"]]
        _, body = crow_body(crow, msgs, doc["served_model"])
        if not body.get("tools") or body.get("reasoning_effort") != "high":
            # crow_core without its manifests/ beside cli/ sends no tools, no reasoning_effort
            # and min_p 0.01: a different prompt (measured 2026-09-23: 36 tokens short, thinking off)
            sys.exit("crow_core built a body without tools / reasoning_effort high - its manifests/ "
                     "directory is missing beside cli/ (%s)" % crow_core)
        b = forced_body(body, rq["force_ids"])
        t0 = time.time()
        try:
            ans = post(url, b)
        except urllib.error.HTTPError as exc:
            sys.exit("segment %s K %d: HTTP %d %s" % (rq["segment"], rq["at"], exc.code, exc.read()[:300]))
        ents = entries_of(ans)
        usage = ans.get("usage") or {}
        rec = {"segment": rq["segment"], "at": rq["at"], "forced": len(rq["force_ids"]),
               "entries": len(ents), "prompt_tokens": usage.get("prompt_tokens"),
               "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
               "seconds": round(time.time() - t0, 1), "timings": ans.get("timings")}
        reqs.append(rec)
        for sid in sids:
            s = by_id[sid]
            j = s["token_index"]
            if j >= len(ents):
                sys.exit("site %s: only %d entries came back (need %d)" % (sid, len(ents), j + 1))
            got = read_site(ents[j], s)
            got["correct_forced"] = False
            if got["lp_correct"] is None:
                # outside the top 20: force the correct id at j and read it exactly
                b2 = forced_body(body, rq["force_ids"][:j] + [s["correct_id"]])
                e2 = entries_of(post(url, b2))
                got["lp_correct"] = e2[j]["logprob"]
                got["correct_forced"] = True
            row = {k: s.get(k) for k in ("id", "segment", "at", "token_index", "class", "context", "produced", "correct",
                                     "corrupt_id", "correct_id", "corrupt_token", "correct_token")}
            row.update(got)
            row["margin"] = round(got["lp_correct"] - got["lp_corrupt"], 6)
            row["corrupt_wins"] = got["lp_corrupt"] > got["lp_correct"]
            row["prompt_tokens_delta_vs_live"] = (rec["prompt_tokens"] - s["live_prompt_tokens"]
                                                  if rec["prompt_tokens"] is not None and s.get("live_prompt_tokens")
                                                  else None)
            rows.append(row)
            print("%-14s lp(correct %r) %8.4f  lp(corrupt %r) %8.4f  margin %+8.4f  top1 %r%s" % (
                sid, s["correct_token"], got["lp_correct"], s["corrupt_token"], got["lp_corrupt"],
                row["margin"], got["top1_token"], "  [correct forced]" if got["correct_forced"] else ""),
                file=sys.stderr)
        print("  seg %s K %d: prompt %s (%s cached, live %s), %d forced, %.1f s" % (
            rq["segment"], rq["at"], rec["prompt_tokens"], rec["cached_tokens"],
            by_id[sids[0]]["live_prompt_tokens"], rec["forced"], rec["seconds"]), file=sys.stderr)
    summ = summarize(rows)
    out = {"label": args.label, "warm_chain": bool(args.warm_chain), "sites_file": os.path.relpath(sites_path, REPO), "url": url,
           "seconds": round(time.time() - t_all, 1), "summary": summ, "rows": rows, "requests": reqs}
    with open(args.json, "w") as fh:
        json.dump(out, fh, indent=1)
        fh.write("\n")
    print("%s: corrupt wins %d/%d (%.3f), mean margin %+.3f nats, correct top-1 %d, %.0f s" % (
        args.label, summ["corrupt_wins"], summ["sites_measured"], summ["corrupt_win_fraction"] or 0,
        summ["mean_margin"] or 0, summ["correct_top1"], out["seconds"]))


# ------------------------------------------------------------------ freerun / speed

DIGIT_ERRORS = [
    ("mat4 index > 15", r"\bo\[(\d+)\]", lambda m: int(m.group(1)) > 15),
    ("identifier digit doubled", r"\b(mat44|mat33|N33|texImage33D|texImage22D|vec44|vec33)\b", None),
    ("ampersand in number list", r"\d&\d", None),
    ("home user not nibor1896", r"/home/(nibor\d+)", lambda m: m.group(1) != "nibor1896"),
    ("markup in value", r"parameter_|</invoke>|<result>", None),
]


def count_digit_errors(text):
    import re
    out, total = {}, 0
    for name, pat, pred in DIGIT_ERRORS:
        hits = [m.group(0) for m in re.finditer(pat, text) if pred is None or pred(m)]
        if hits:
            out[name] = hits
            total += len(hits)
    idx = [int(x) for x in re.findall(r"\bo\[(\d+)\]", text)]
    return total, out, len(idx)


def freerun(args):
    with open(args.sites) as fh:
        doc = json.load(fh)
    spec_dir = os.path.dirname(resolve(doc["spec"], REPO))
    crow = load_crow(resolve(doc["crow_core"], spec_dir))
    sessions = {seg: json.load(open(resolve(s["file"], spec_dir))) for seg, s in doc["segments"].items()}
    url = (args.base_url or "http://127.0.0.1:%d/v1" % args.port).rstrip("/") + "/chat/completions"
    res = []
    for fr in doc.get("freerun", []):
        _, body = crow_body(crow, sessions[fr["segment"]]["messages"][:fr["at"]], doc["served_model"])
        b = forced_body(body, fr["force_ids"])
        b["max_tokens"] = len(fr["force_ids"]) + fr.get("max_new", 600)
        b["logprobs"] = False
        b.pop("top_logprobs", None)
        t0 = time.time()
        ans = post(url, b)
        msg = ans["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        text = "".join((c.get("function") or {}).get("arguments") or "" for c in calls) or (msg.get("content") or "")
        # the arguments JSON escapes newlines; the errors are counted on the decoded value
        try:
            val = json.loads(calls[0]["function"]["arguments"]).get(fr["param"], "") if calls else text
        except (json.JSONDecodeError, KeyError, IndexError):
            val = text
        n, kinds, nidx = count_digit_errors(val)
        tm = ans.get("timings") or {}
        r = {"id": fr["id"], "segment": fr["segment"], "at": fr["at"], "forced": len(fr["force_ids"]),
             "digit_errors": n, "kinds": kinds, "o_indices": nidx, "chars": len(val),
             "finish": ans["choices"][0].get("finish_reason"), "usage": ans.get("usage"),
             "timings": tm, "seconds": round(time.time() - t0, 1), "value": val}
        res.append(r)
        print("%s freerun %-12s: %d digit errors in %d chars (%d o[] indices) %s" % (
            args.label, fr["id"], n, len(val), nidx, json.dumps(kinds)[:300]), file=sys.stderr)
    with open(args.json, "w") as fh:
        json.dump({"label": args.label, "freerun": res}, fh, indent=1)
        fh.write("\n")
    print("%s freerun: %d digit errors over %d points" % (args.label, sum(r["digit_errors"] for r in res), len(res)))


SPEED_PROMPT = ("Write a detailed, plain-prose description of how a lighthouse keeper's day went in 1890, "
                "hour by hour, in about 400 words.")


def speed(args):
    url = (args.base_url or "http://127.0.0.1:%d/v1" % args.port).rstrip("/") + "/chat/completions"
    out = []
    for i in range(args.repeat):
        b = {"model": "crow", "messages": [{"role": "user", "content": SPEED_PROMPT}], "temperature": 0,
             "max_tokens": args.tokens, "stream": False}
        t0 = time.time()
        ans = post(url, b)
        tm = ans.get("timings") or {}
        u = ans.get("usage") or {}
        out.append({"completion_tokens": u.get("completion_tokens"), "predicted_per_second": tm.get("predicted_per_second"),
                    "predicted_ms": tm.get("predicted_ms"), "wall_s": round(time.time() - t0, 2)})
        print("%s speed %d: %s tok, %s tok/s (serve timings)" % (args.label, i, u.get("completion_tokens"),
                                                                tm.get("predicted_per_second")), file=sys.stderr)
    tps = [x["predicted_per_second"] for x in out if x["predicted_per_second"]]
    doc = {"label": args.label, "runs": out, "median_tok_s": sorted(tps)[len(tps) // 2] if tps else None}
    with open(args.json, "w") as fh:
        json.dump(doc, fh, indent=1)
        fh.write("\n")
    print("%s speed: median %.2f tok/s over %d runs of %d tokens" % (args.label, doc["median_tok_s"] or 0,
                                                                   len(tps), args.tokens))


# ------------------------------------------------------------------ table

def table(args):
    arms = []
    for p in args.arms:
        with open(p) as fh:
            arms.append(json.load(fh))
    ref = arms[0]
    lines = ["| arm | corrupt wins | fraction | mean margin (nats) | median margin | mean p(corrupt) at T=1 | correct top-1 | seconds |",
             "|---|---|---|---|---|---|---|---|"]
    for a in arms:
        s = a["summary"]
        pc = sum(math.exp(r["lp_corrupt"]) for r in a["rows"]) / len(a["rows"])
        lines.append("| %s | %d/%d | %.3f | %+.3f | %+.3f | %.3f | %d | %.0f |" % (
            a["label"], s["corrupt_wins"], s["sites_measured"], s["corrupt_win_fraction"],
            s["mean_margin"], s["median_margin"], pc, s["correct_top1"], a["seconds"]))
    ctx = {x["id"]: x.get("context") for x in ref["rows"]}
    if any(ctx.values()):
        lines += ["", "Fresh sites only (the corrupt spelling is not yet in the context):", "",
                  "| arm | corrupt wins (fresh) | mean margin (fresh) | mean p(corrupt) at T=1 (fresh) |", "|---|---|---|---|"]
        for a in arms:
            fr = [r for r in a["rows"] if ctx.get(r["id"]) == "fresh"]
            if fr:
                lines.append("| %s | %d/%d | %+.3f | %.3f |" % (
                    a["label"], sum(r["corrupt_wins"] for r in fr), len(fr),
                    sum(r["margin"] for r in fr) / len(fr), sum(math.exp(r["lp_corrupt"]) for r in fr) / len(fr)))
    lines += ["", "Per site: margin = lp(correct) - lp(corrupt) in nats (negative = the corrupt token wins).", ""]
    lines.append("| site | context | correct / corrupt | " + " | ".join(a["label"] for a in arms) + " |")
    lines.append("|---|---|---|" + "---|" * len(arms))
    for i, r in enumerate(ref["rows"]):
        cells = []
        for a in arms:
            x = next((y for y in a["rows"] if y["id"] == r["id"]), None)
            cells.append("n/a" if x is None else "%+.2f%s" % (x["margin"], " W" if x["corrupt_wins"] else ""))
        lines.append("| %s | %s | %r / %r | %s |" % (r["id"], ctx.get(r["id"]) or "", r["correct_token"], r["corrupt_token"], " | ".join(cells)))
    lines.append("")
    lines.append("W = the corrupt token beats the correct one in that arm.")
    if len(arms) > 1:
        lines += ["", "Sites fixed vs %s (corrupt wins there, not in the arm):" % ref["label"], ""]
        refw = {r["id"] for r in ref["rows"] if r["corrupt_wins"]}
        for a in arms[1:]:
            aw = {r["id"] for r in a["rows"] if r["corrupt_wins"]}
            lines.append("- %s: fixed %d (%s), newly broken %d (%s)" % (
                a["label"], len(refw - aw), ", ".join(sorted(refw - aw)) or "-",
                len(aw - refw), ", ".join(sorted(aw - refw)) or "-"))
    text = "\n".join(lines) + "\n"
    if args.md:
        with open(args.md, "w") as fh:
            fh.write(text)
    print(text)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--spec", required=True)
    b.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--sites", required=True)
    r.add_argument("--port", type=int, default=8111)
    r.add_argument("--base-url", default=None)
    r.add_argument("--label", default="arm")
    r.add_argument("--only", default=None, help="comma-separated site ids")
    r.add_argument("--warm-chain", action="store_true",
                   help="send every earlier turn's prompt of the segment first (the live WARM prefill path)")
    r.add_argument("--crow-core", default=None)
    r.add_argument("--json", required=True)
    f = sub.add_parser("freerun")
    f.add_argument("--sites", required=True)
    f.add_argument("--port", type=int, default=8111)
    f.add_argument("--base-url", default=None)
    f.add_argument("--label", default="arm")
    f.add_argument("--json", required=True)
    v = sub.add_parser("speed")
    v.add_argument("--port", type=int, default=8111)
    v.add_argument("--base-url", default=None)
    v.add_argument("--label", default="arm")
    v.add_argument("--tokens", type=int, default=256)
    v.add_argument("--repeat", type=int, default=3)
    v.add_argument("--json", required=True)
    t = sub.add_parser("table")
    t.add_argument("arms", nargs="+", help="arm JSONs; the first is the reference")
    t.add_argument("--md", default=None)
    args = ap.parse_args(argv)
    return {"build": build, "run": run, "table": table, "freerun": freerun, "speed": speed}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
