#!/usr/bin/env python3
"""#68 step 2: the long-context quality gate - one AGENTIC session shape at a chosen context
size, greedy, scored the way `docs/ten-tasks.md` scores (Pass / Fail per probe, and any token
degeneration is a Fail whatever else the answer said).

Usage, from anywhere, against a `serve` started by `tools/serve-linux.sh`:
  tools/longctx-gate.py --target-tokens 100000 --out decode_out/longctx-100k.json
  tools/longctx-gate.py --target-tokens 170000 --out decode_out/longctx-170k.json

Why this exists: the ten-task gate never measured anything past 16k of context, so before #68
there was no quality reading of this engine's decode at 100k+ at all. This is that reading, in
one reproducible command, with its expected values recorded in
`docs/long-context-goalmode.md`.

The session shape (the same shape Crow drives, without Crow):

- One system turn, then N rounds of `assistant` tool call -> `tool` result, each result one
  synthetic source file of the `libghost` crate. That is the shape of an agent that read a repo:
  the context is tool output, not one pasted document.
- The material is GENERATED here, deterministically, from the file index alone - no repo file is
  read, so the gate cannot drift when the tree changes, and the same command gives the same
  bytes on any machine. Three unique constants are planted at ~5 %, ~50 % and ~95 % of the
  material, and one function contradicts its own doc comment at ~70 %.
- Then five probe turns, each sent as the next user turn of the SAME session, so the engine
  answers them the way it answers a real turn: whole history re-sent, prefix cache warm, greedy.

The five probes and what a Pass needs (all substrings of one probe must be present, case
insensitive, and the answer must not degenerate):

| # | probe | class | Pass needs |
|---|---|---|---|
| 1 | the constant planted at ~5 % depth | retrieval, early | its value and its file name |
| 2 | the constant planted at ~50 % depth | retrieval, middle | its value and its file name |
| 3 | the constant planted at ~95 % depth | retrieval, late | its value and its file name |
| 4 | add two of them, subtract the third | multi-step over three depths | the exact number |
| 5 | the one function whose body contradicts its doc | debug / synthesis | the function name, its file, and the wrong operator |

Degeneration (a Fail on its own, `docs/ten-task-expected.md` §1): one unit repeated over half
the answer with at least 20 repeats, or one word over half the words.

Exit 0 when the run reached the recorded expectation (`--want-pass`, default 4 of 5 with 0
degenerate), 1 when it did not, 2 on a usage error or a refused request.
"""
import argparse
import collections
import json
import os
import sys
import time
import urllib.error
import urllib.request

SYSTEM = ("You are a local coding assistant. You have already read the files below with your "
          "tools. Answer from that material only, name the file you took each fact from, and "
          "keep every answer under 120 words.")

# the three planted constants: name, value, depth in the material (fraction)
PLANTED = [("RING_FLOOR", 5137, 0.05), ("TILE_SPAN", 9281, 0.50), ("PURGE_MARK", 4409, 0.95)]
# the planted contradiction: its function name, the wrong operator, its depth
BUG = ("checksum_rows", "-=", 0.70)


def file_text(i, planted=None, bug=False):
    """one synthetic source file, a pure function of its index - byte-stable everywhere"""
    a, b, c = 17 + i * 3, 29 + i * 7, 41 + i * 11
    out = [f"// libghost/src/f{i:03d}.rs - stage {i} of the ghost pipeline.",
           f"// Layer group {i % 7}, tile span {b}, ring rows {c}.",
           "",
           f"pub const STAGE_{i:03d}_WIDTH: usize = {a * 13};",
           f"pub const STAGE_{i:03d}_DEPTH: usize = {b * 3};",
           ""]
    if planted:
        name, value = planted
        out += [f"/// The one {name} of the crate; every stage reads it and none writes it.",
                f"pub const {name}: usize = {value};", ""]
    for f in range(6):
        n = i * 6 + f
        out += [f"/// Fold stage {i} band {f} into the accumulator and return its mean.",
                f"pub fn fold_{n:04d}(rows: &mut [f32], scale: f32) -> f32 {{",
                "    let mut acc = 0.0f32;",
                "    for (j, r) in rows.iter_mut().enumerate() {",
                f"        let w = if j % {3 + f} == 0 {{ scale }} else {{ scale * 0.5 }};",
                f"        *r = r.mul_add(w, {a}.0);",
                "        acc += *r;",
                "    }",
                "    acc / (rows.len() as f32).max(1.0)",
                "}",
                ""]
    if bug:
        out += ["/// Return the MEAN of the rows: the sum of every row divided by their count.",
                "pub fn checksum_rows(rows: &[f32]) -> f32 {",
                "    let mut acc = 0.0f32;",
                "    for r in rows {",
                "        acc -= *r;",
                "    }",
                "    acc / (rows.len() as f32).max(1.0)",
                "}",
                ""]
    return "\n".join(out)


def build_material(n_files):
    """(files, names) - `files[i]` is the text of `libghost/src/f<i>.rs`"""
    marks = {}
    for name, value, depth in PLANTED:
        marks[min(n_files - 1, int(depth * n_files))] = (name, value)
    bug_at = min(n_files - 1, int(BUG[2] * n_files))
    files = [file_text(i, marks.get(i), i == bug_at) for i in range(n_files)]
    where = {name: f"f{i:03d}.rs" for i, (name, _v) in marks.items()}
    where[BUG[0]] = f"f{bug_at:03d}.rs"
    return files, where


def build_session(files):
    """the agentic history: one tool call plus one tool result per file"""
    msgs = [{"role": "system", "content": SYSTEM}]
    for i, text in enumerate(files):
        name = f"libghost/src/f{i:03d}.rs"
        msgs.append({"role": "user", "content": f"Read {name} and keep it in mind."})
        msgs.append({"role": "assistant", "content": "",
                     "tool_calls": [{"id": f"call_{i}", "type": "function",
                                     "function": {"name": "read_file",
                                                  "arguments": json.dumps({"path": name})}}]})
        msgs.append({"role": "tool", "content": text, "tool_call_id": f"call_{i}"})
        msgs.append({"role": "assistant", "content": f"Read {name}."})
    return msgs


def probes(where):
    """(question, [required substrings]) per probe - the recorded Pass condition"""
    r, t, p = (v for _n, v, _d in PLANTED)
    return [
        (f"What is the value of the constant RING_FLOOR, and which file declares it?",
         [str(r), where["RING_FLOOR"]]),
        (f"What is the value of the constant TILE_SPAN, and which file declares it?",
         [str(t), where["TILE_SPAN"]]),
        (f"What is the value of the constant PURGE_MARK, and which file declares it?",
         [str(p), where["PURGE_MARK"]]),
        ("Take RING_FLOOR plus TILE_SPAN minus PURGE_MARK. Give the resulting number and show "
         "the three values you used.", [str(r + t - p)]),
        ("Exactly one function in the material does not do what its own doc comment says. "
         "Name the function, name its file, and name the operator that is wrong.",
         [BUG[0], where[BUG[0]]]),
    ]


def degenerate(answer):
    t = answer.strip()
    best = 0
    for u in range(1, 9):
        for i in range(len(t) - u + 1):
            unit = t[i:i + u]
            reps = 1
            while t[i + reps * u:i + (reps + 1) * u] == unit:
                reps += 1
            if reps >= 3:
                best = max(best, reps * u)
    words = t.split()
    top = collections.Counter(words).most_common(1)
    wf = (top[0][1] / len(words)) if words else 0.0
    rf = (best / len(t)) if t else 0.0
    return bool((rf >= 0.5 and best >= 20) or (wf >= 0.5 and len(words) >= 20)), round(rf, 3), round(wf, 3)


def stream_round(base, messages, max_tokens, timeout):
    body = {"model": "crow-nest", "messages": messages, "stream": True,
            "stream_options": {"include_usage": True}, "timings_per_token": True,
            "max_tokens": max_tokens, "temperature": 0}
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    content, reasoning, finish, usage, timings = "", "", None, {}, {}
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            line = line.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            frame = json.loads(payload)
            usage = frame.get("usage") or usage
            timings = frame.get("timings") or timings
            choice = (frame.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            if delta.get("content"):
                content += delta["content"]
            if delta.get("reasoning_content"):
                reasoning += delta["reasoning_content"]
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
    return content, reasoning, finish, usage, timings, round(time.time() - t0, 2)


def main():
    sys.stdout.reconfigure(line_buffering=True)   # a 3-minute prefill must not sit in a buffer
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--target-tokens", type=int, default=100000)
    ap.add_argument("--ratio", type=float, default=2.44,
                    help="chars per token of this synthetic source (measured 2026-09-18: "
                         "23,910 chars -> 9,830 ids through `serve tokenize --chat`)")
    ap.add_argument("--wrap-tokens", type=int, default=60,
                    help="ids the four history messages around one file add (estimate)")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--want-pass", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    per_file = len(file_text(0)) / a.ratio + a.wrap_tokens
    n_files = max(4, int(a.target_tokens / per_file))
    files, where = build_material(n_files)
    chars = sum(len(f) for f in files)
    msgs = build_session(files)
    print(f"longctx-gate.py: {n_files} synthetic files, {chars} chars of material, "
          f"~{int(chars / a.ratio)} tokens estimated, {len(msgs)} history messages")
    print(f"longctx-gate.py: planted {where}")

    base = f"http://127.0.0.1:{a.port}"
    record = {"target_tokens": a.target_tokens, "n_files": n_files, "material_chars": chars,
              "history_messages": len(msgs), "planted": where, "sampling": "greedy",
              "max_tokens": a.max_tokens, "probes": []}
    history = list(msgs)
    for i, (q, need) in enumerate(probes(where)):
        history.append({"role": "user", "content": q})
        try:
            content, reasoning, finish, usage, timings, wall = stream_round(
                base, history, a.max_tokens, a.timeout)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:600]
            print(f"probe {i + 1}: HTTP {exc.code} -> {detail}")
            record["probes"].append({"probe": i + 1, "http": exc.code, "error": detail})
            record["summary"] = {"passed": 0, "of": len(probes(where)), "refused": True}
            if a.out:
                json.dump(record, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            return 2
        low = content.lower()
        missing = [s for s in need if s.lower() not in low]
        deg, rf, wf = degenerate(content)
        verdict = "Pass" if (not missing and not deg) else "Fail"
        row = {"probe": i + 1, "question": q, "need": need, "missing": missing,
               "verdict": verdict, "degenerate": deg, "repeat_frac": rf, "top_word_frac": wf,
               "prompt_tokens": usage.get("prompt_tokens"),
               "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
               "prefilled": timings.get("prompt_n"),
               "prompt_per_second": timings.get("prompt_per_second"),
               "completion_tokens": usage.get("completion_tokens"),
               "decode_per_second": timings.get("predicted_per_second"),
               "finish": finish, "wall_s": wall, "answer": content, "reasoning": reasoning}
        record["probes"].append(row)
        print(f"probe {i + 1}: {verdict}  prompt {row['prompt_tokens']} tok "
              f"({row['cached_tokens']} cached, {row['prefilled']} prefilled at "
              f"{row['prompt_per_second']} tok/s), {row['completion_tokens']} tok at "
              f"{row['decode_per_second']} tok/s, wall {wall}s, missing {missing}, "
              f"degenerate {deg}")
        print(f"        answer {content[:260]!r}")
        history.append({"role": "assistant", "content": content})

    passed = sum(1 for p in record["probes"] if p.get("verdict") == "Pass")
    degs = sum(1 for p in record["probes"] if p.get("degenerate"))
    record["summary"] = {"passed": passed, "of": len(record["probes"]), "degenerate": degs,
                         "want_pass": a.want_pass}
    print(f"longctx-gate.py: {passed} of {len(record['probes'])} Pass, {degs} degenerate "
          f"(want >= {a.want_pass} Pass and 0 degenerate)")
    if a.out:
        json.dump(record, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"longctx-gate.py: record written to {a.out}")
    return 0 if (passed >= a.want_pass and degs == 0) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
