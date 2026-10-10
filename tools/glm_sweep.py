#!/usr/bin/env python3
"""GLM-5.3-Flash speed sweep against engine `serve`, measured the way 0xSero's sweep measures sglang.

Template: github.com/sybil-solutions/glm53-flash-offload at 6769b27, `bench/sweep.py` ("protocol v1"), run there as
`--template glm --prefill 8192 32768 --conc 1 2 4 --reps 3 --dec-reps 2`. Same order, same arithmetic, same JSON keys:

  1. warm-up: one 512-token request, not recorded (here it also calibrates the template overhead, see below)
  2. PREFILL per size n: REPS fresh random prompts; tok/s = n / TTFT, TTFT = first SSE data frame - request start;
     median + min/max/all of the REPS.
  3. DECODE per concurrency C: C simultaneous greedy requests with distinct short prompts (his TOPICS and nonce text,
     variant = round * 100 + i), run to natural EOS, DEC_REPS rounds per C.
     aggregate = all completion tokens / (last token time - first first-token time);
     per-stream = (tokens - 1) / (last - first) per request; median over the rounds.
  4. C1@32k: C1 with a 32,768-token random prefix in front of the chat prompt, when n_ctx > 40,000.

What serve cannot do the way sglang does, and what this script does instead (all recorded in the JSON):

- serve has no raw-ids route (`/generate` + `input_ids`) and no `/tokenize`; it renders every prompt with the GLM chat
  template. Prefill prompts are therefore random single-token vocabulary words (byte-level `Ġ` + letters, any script,
  ids in his range [1000, 150000), read from `tokenizer.json`) inside one user message. The warm-up measures the
  template's overhead from `usage.prompt_tokens`, so a prefill prompt is n tokens IN TOTAL; the rate uses the
  `prompt_tokens` serve reports. His prompts are raw random ids with no template.
- His decode prompt is `[gMASK]<sop><|user|>{q}<|assistant|><think></think>` (thinking off, no system line). GLM's
  template cannot render thinking off (docs/glm5-tokenizer.md); serve renders `<|system|>Reasoning Effort: Max ...
  <|assistant|><think>`. `reasoning_budget_tokens: 0` closes the block before it opens (serve forces `</think>` and
  `\\n\\n` as the first two generated ids), so the answer is a thinking-off answer as in his run.
- TTFT: his first `data:` line carries the first token; serve's first frame (`delta.role`) leaves after the prefill
  AND the first id (`glm_generate`: `sink.open` follows `prefill`), so the first data frame is the same instant.
  Prefill requests send `max_tokens: 1` instead of hanging up after the first frame.
- `max_tokens`: his decode budget is the remaining context (uncapped); serve caps at 32,768. Answers run ~1.5-2.3k
  tokens, `finish` records whether a cap was hit.
- Concurrency: his sglang ran with `--max-batch-size 8` (batched C2/C4). The glm5_next serve loop is
  "blocking, one request at a time": C2/C4 requests queue in the listen backlog and run one after another.
  Every round records `serial` (no two streams overlapped); his rows of that kind say "(serial)".
- No best-known gates / early exit (his `best_known.json` is his card's).

Run:   python tools/glm_sweep.py --url http://127.0.0.1:8099 --config "<commit + env>" --out <sweep.json> \\
           --tokenizer <models/GLM-5.3-Flash-original/tokenizer.json> --prefill 8192 32768 --conc 1 2 4 --reps 3 --dec-reps 2
Self-test (no GPU, no model, a mock serve on a free port):   python tools/glm_sweep.py --self-test
"""
import argparse
import hashlib
import http.server
import json
import os
import random
import secrets
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TOKENIZER = os.path.join(REPO, "models", "GLM-5.3-Flash-original", "tokenizer.json")
SERVE_MAX_TOKENS = 32768  # serve's cap on max_tokens (serve.rs MAX_MAX_TOKENS)
# verbatim from his bench/sweep.py
TOPICS = ["the history of the printing press", "how a jet engine works", "the causes of the French revolution",
          "photosynthesis in C4 plants", "how TCP congestion control works", "the life cycle of stars",
          "how vaccines train the immune system", "the economics of container shipping", "how compilers optimise loops",
          "the geology of volcanoes", "the rules and strategy of chess openings", "how GPS determines position",
          "the history of the bicycle", "how noise-cancelling headphones work", "the water cycle",
          "how databases implement transactions"]

NONCE = secrets.token_hex(16)
PROMPT_HASHES = []


# ------------------------------------------------------------------ prompts

def bytes_to_unicode():
    """the byte-level BPE alphabet (GPT-2's table, which `ByteLevel` in tokenizer.json uses)"""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) + list(range(ord("\xae"), ord("\xff") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def vocab_words(tokenizer_path):
    """every vocabulary entry with id in [1000, 150000) that is one space plus letters only (`\\p{L}`), decoded:
    the pre-tokenizer keeps ` word` as one piece and the vocabulary holds it whole, so one word is one token"""
    t = json.load(open(tokenizer_path, encoding="utf-8"))
    inv = {c: b for b, c in bytes_to_unicode().items()}
    out = []
    for tok, i in t["model"]["vocab"].items():
        if not (1000 <= i < 150000) or not tok.startswith("Ġ"):
            continue
        try:
            w = bytes(inv[c] for c in tok).decode("utf-8")
        except (KeyError, UnicodeDecodeError):
            continue
        if len(w) >= 2 and w[0] == " " and w[1:].isalpha():
            out.append((i, w))
    out.sort()
    return [w for _, w in out]


def rand_words(words, k, seed):
    rng = random.Random(seed)
    return "".join(rng.choice(words) for _ in range(max(k, 1)))


def chat_text(i):
    """his `chat_ids` question, verbatim (the template around it is serve's)"""
    q = f"[cache-bust nonce: {NONCE}-{i}]\nWrite a detailed, well-structured explanation of {TOPICS[i % len(TOPICS)]} (variant {i})."
    PROMPT_HASHES.append(hashlib.sha256(q.encode()).hexdigest())
    return q


# ------------------------------------------------------------------ HTTP

def post(url, payload, timeout=7200):
    """one streamed chat completion: (t0, data-frame times, final chunk, content text)"""
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    times, last, text = [], None, []
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    break
                last = json.loads(body)
                times.append(time.perf_counter())
                for ch in last.get("choices") or []:
                    text.append((ch.get("delta") or {}).get("content") or "")
    except urllib.error.HTTPError as e:
        raise SystemExit(f"serve answered {e.code}: {e.read().decode(errors='replace')[:500]}")
    if not times:
        raise SystemExit("serve sent no data frame")
    return t0, times, last, "".join(text)


def final_fields(last):
    usage = (last or {}).get("usage") or {}
    ch = ((last or {}).get("choices") or [{}])[0]
    return {"prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "finish": ch.get("finish_reason"), "timings": (last or {}).get("timings")}


def body(content, max_tokens, think_off):
    b = {"messages": [{"role": "user", "content": content}], "stream": True, "temperature": 0,
         "max_tokens": max_tokens, "stream_options": {"include_usage": True}, "timings_per_token": True}
    if think_off:
        b["reasoning_budget_tokens"] = 0
    return b


class Power:
    """nvidia-smi power.draw sampled every 0.25 s while active (his class, unchanged)"""
    def __init__(self):
        self.samples, self.on = [], False

    def __enter__(self):
        self.on = True

        def run():
            while self.on:
                try:
                    o = subprocess.run(["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                                       capture_output=True, text=True, timeout=5).stdout.split()
                    if o:
                        self.samples.append(sum(float(x) for x in o))
                except Exception:
                    pass
                time.sleep(0.25)
        self.t = threading.Thread(target=run, daemon=True)
        self.t.start()
        return self

    def __exit__(self, *a):
        self.on = False
        self.t.join(timeout=6)

    def mean(self):
        return round(sum(self.samples) / len(self.samples), 1) if self.samples else None


def med(xs):
    return {"median": statistics.median(xs), "min": min(xs), "max": max(xs), "n": len(xs), "all": xs}


def error_row(res, key, rounds, power):
    """a decode cell with a failed round: no rate, the rounds as received, the error named in res["errors"]"""
    errs = [r["error"] for r in rounds if "error" in r]
    res.setdefault("errors", []).extend(errs)
    print(f"decode {key}: ERROR {errs[0]}" + (f" (+{len(errs) - 1} more)" if len(errs) > 1 else ""), flush=True)
    return {"error": errs[0], "rounds": rounds, "gpu_power_w_mean": power}


# ------------------------------------------------------------------ the sweep

def main(argv=None, words=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8099")
    ap.add_argument("--card", default=None, help="label, e.g. rtx5090")
    ap.add_argument("--config", required=True, help="label: commit + env summary")
    ap.add_argument("--prefill", type=int, nargs="*", default=[8192, 32768])
    ap.add_argument("--conc", type=int, nargs="*", default=[1, 2, 4])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--dec-reps", type=int, default=2)
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER, help="GLM-5.3-Flash tokenizer.json (prefill words)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--power", action=argparse.BooleanOptionalAction, default=True, help="sample nvidia-smi power")
    a = ap.parse_args(argv)
    if words is None:
        words = vocab_words(a.tokenizer)
    props = json.loads(urllib.request.urlopen(a.url + "/props", timeout=30).read())
    ctx_len = int(props.get("n_ctx") or 131072)
    res = {"card": a.card, "config": a.config, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "ctx_len": ctx_len,
           "protocol": "bench/sweep.py v1 (sybil-solutions/glm53-flash-offload 6769b27) via tools/glm_sweep.py",
           "api": "crow-nest serve /v1/chat/completions", "served_model": props.get("model"), "template": "glm",
           "server_props": props, "prefill_prompt": f"random single-token vocabulary words ({len(words)} candidates)",
           "decode_prompt": "his question text; serve's GLM template; reasoning_budget_tokens 0",
           "prefill": {}, "decode": {}, "status": "RUNNING"}

    def save():
        json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    seed = int(time.time())
    # warm-up (his: 512 random ids, first token only); here also the template overhead
    _, _, last, _ = post(a.url, body(rand_words(words, 512, seed), 1, False))
    pt = final_fields(last)["prompt_tokens"]
    overhead = (pt - 512) if pt is not None else 0
    res["template_overhead_tokens"] = overhead
    print(f"warm-up: {pt} prompt tokens, template overhead {overhead}", flush=True)

    receipts = []
    for n in a.prefill:
        if n + 64 > ctx_len:
            res["prefill"][str(n)] = {"skipped": f"> context {ctx_len}"}
            continue
        rates = []
        for rep in range(a.reps):
            content = rand_words(words, n - overhead, seed + 1000 * n + rep)
            t0, times, last, _ = post(a.url, body(content, 1, False))
            f = final_fields(last)
            tokens = f["prompt_tokens"] if f["prompt_tokens"] is not None else n
            ttft = times[0] - t0
            rates.append(round(tokens / ttft, 1))
            receipts.append({"tokens": n, "rep": rep, "prompt_tokens": f["prompt_tokens"], "cached_tokens": f["cached_tokens"],
                             "prompt_sha256": hashlib.sha256(content.encode()).hexdigest(), "ttft_s": ttft,
                             "server_timings": f["timings"]})
            print(f"prefill {n}: {rates[-1]} tok/s ({tokens} prompt tokens)", flush=True)
        res["prefill"][str(n)] = med(rates)
        res["prefill_prompt_receipts"] = receipts
        save()

    def run_conc(c, rnd, prefix=None):
        out = [None] * c

        def one(i):
            q = chat_text(rnd * 100 + i)
            content = prefix + "\n\n" + q if prefix else q
            t0, times, last, text = post(a.url, body(content, SERVE_MAX_TOKENS, True))
            f = final_fields(last)
            n = f["completion_tokens"] if f["completion_tokens"] is not None else len(times)
            out[i] = {"t0": t0, "first": times[0], "last": times[-1], "tokens": n, "finish": f["finish"], "response": text,
                      "prompt_tokens": f["prompt_tokens"], "cached_tokens": f["cached_tokens"], "server_timings": f["timings"],
                      "frames": len(times)}
        ths = [threading.Thread(target=one, args=(i,)) for i in range(c)]
        [t.start() for t in ths]
        [t.join() for t in ths]
        if any(o is None for o in out):
            raise SystemExit("a decode stream failed")
        tot = sum(o["tokens"] for o in out)
        span = max(o["last"] for o in out) - min(o["first"] for o in out)
        per = [(o["tokens"] - 1) / (o["last"] - o["first"]) for o in out if o["tokens"] > 1 and o["last"] > o["first"]]
        if not per or span <= 0:
            # no stream got past its first frame (serve died after the prefill, 2026-10-10): an error
            # row with the streams as received, not a rate
            return {"error": f"zero decode tokens in C{c} round {rnd}: no stream sent a token after its first frame "
                             f"(finish {sorted({str(o['finish']) for o in out})}, frames {[o['frames'] for o in out]})",
                    "tokens": tot, "streams": out}
        by_first = sorted(out, key=lambda o: o["first"])
        serial = c > 1 and all(b["first"] >= a_["last"] for a_, b in zip(by_first, by_first[1:]))
        return {"aggregate": round(tot / span, 2), "per_stream_mean": round(statistics.mean(per), 2),
                "per_stream_min": round(min(per), 2), "tokens": tot,
                "finish": sorted({o["finish"] for o in out if o["finish"]}),
                "ttft_max": round(max(o["first"] - o["t0"] for o in out), 2), "serial": serial, "streams": out}

    for c in a.conc:
        if a.power:
            with Power() as pw:
                rounds = [run_conc(c, r) for r in range(a.dec_reps)]
            power = pw.mean()
        else:
            rounds, power = [run_conc(c, r) for r in range(a.dec_reps)], None
        if any("error" in r for r in rounds):
            res["decode"][f"C{c}"] = error_row(res, f"C{c}", rounds, power)
            save()
            continue
        agg = [r["aggregate"] for r in rounds]
        res["decode"][f"C{c}"] = {"aggregate": med(agg), "per_stream_mean": med([r["per_stream_mean"] for r in rounds]),
                                  "rounds": rounds, "gpu_power_w_mean": power,
                                  "serial": c > 1 and all(r["serial"] for r in rounds)}
        print(f"decode C{c}: aggregate {statistics.median(agg):.2f} tok/s, per-stream "
              f"{statistics.median([r['per_stream_mean'] for r in rounds]):.2f}"
              f"{' (serial)' if res['decode'][f'C{c}']['serial'] else ''}", flush=True)
        save()
    if ctx_len > 40000:
        pre = rand_words(words, 32768, seed + 99)
        r = [run_conc(1, 90 + i, prefix=pre) for i in range(a.dec_reps)]
        if any("error" in x for x in r):
            res["decode"]["C1@32k"] = error_row(res, "C1@32k", r, None)
        else:
            res["decode"]["C1@32k"] = {"aggregate": med([x["aggregate"] for x in r]), "rounds": r}
            print(f"decode C1@32k: {statistics.median([x['aggregate'] for x in r]):.2f} tok/s", flush=True)
    res["prompt_sha256"] = PROMPT_HASHES
    res["nonce_prefix"] = NONCE
    res["status"] = "DONE WITH ERRORS" if res.get("errors") else "DONE"
    res["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    save()
    print("SWEEP DONE", a.out, flush=True)
    return res


# ------------------------------------------------------------------ self-test: a mock serve

MOCK_OVERHEAD = 9       # tokens the mock's "template" adds around the user content
MOCK_PREFILL_S = 2e-6   # seconds per prompt token
MOCK_TOKEN_S = 0.002    # seconds per generated token


class MockServe(http.server.BaseHTTPRequestHandler):
    """serve's wire for the routes the sweep uses, one request at a time (HTTPServer is single-threaded, as serve's
    glm5_next loop): /props, and /v1/chat/completions as SSE with role frame after the prefill, content frames,
    a final chunk with usage and timings, [DONE]. One word of content = one token."""
    bodies = []
    protocol_version = "HTTP/1.0"
    # serve's crash of 2026-10-10: a decode request gets its role frame, then the stream ends
    dead_decode = False

    def log_message(self, *a):
        pass

    def do_GET(self):
        doc = json.dumps({"n_ctx": 65536, "model": "mock-glm", "prompt_chunk": 32}).encode()
        self.send_response(200 if self.path == "/props" else 404)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(doc)

    def do_POST(self):
        b = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        MockServe.bodies.append(b)
        content = b["messages"][0]["content"]
        pt = MOCK_OVERHEAD + len(content.split())
        n = 1 if b["max_tokens"] == 1 else 40 + int(hashlib.sha256(content.encode()).hexdigest(), 16) % 20
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        t = time.perf_counter()
        time.sleep(pt * MOCK_PREFILL_S)
        prompt_ms = (time.perf_counter() - t) * 1e3

        def frame(delta, finish=None, extra=None):
            d = {"object": "chat.completion.chunk", "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            d.update(extra or {})
            self.wfile.write(f"data: {json.dumps(d)}\n\n".encode())
            self.wfile.flush()
        frame({"role": "assistant"})
        if MockServe.dead_decode and b["max_tokens"] != 1:
            return
        for i in range(n - 1):
            time.sleep(MOCK_TOKEN_S)
            frame({"content": f" w{i}"})
        frame({}, "stop" if n > 1 else "length",
              {"usage": {"prompt_tokens": pt, "completion_tokens": n, "total_tokens": pt + n,
                         "prompt_tokens_details": {"cached_tokens": 0}},
               "timings": {"prompt_n": pt, "prompt_ms": prompt_ms}})
        self.wfile.write(b"data: [DONE]\n\n")


def self_test():
    srv = http.server.HTTPServer(("127.0.0.1", 0), MockServe)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    words = [f" x{i}" for i in range(500)]
    out = os.path.join(tempfile.mkdtemp(), "sweep.json")
    try:
        main(["--url", url, "--config", "self-test", "--out", out, "--prefill", "256", "1024", "--conc", "1", "2",
              "--reps", "3", "--dec-reps", "2", "--no-power"], words=words)
    finally:
        srv.shutdown()
    r = json.load(open(out, encoding="utf-8"))
    fails = []

    def check(cond, what):
        if not cond:
            fails.append(what)
    check(r["status"] == "DONE", "status DONE")
    check(r["template_overhead_tokens"] == MOCK_OVERHEAD, "warm-up calibrates the template overhead")
    for n in ("256", "1024"):
        p = r["prefill"][n]
        check(p["n"] == 3 and p["median"] == statistics.median(p["all"]), f"prefill {n}: median of 3")
    rc = r["prefill_prompt_receipts"]
    check(all(x["prompt_tokens"] == x["tokens"] for x in rc), "every prefill prompt is n tokens in total")
    check(all(abs(x["tokens"] / x["ttft_s"] - r["prefill"][str(x["tokens"])]["all"][x["rep"]]) <= 0.06 for x in rc),
          "prefill rate = tokens / TTFT")
    check(len({x["prompt_sha256"] for x in rc}) == 6, "a fresh prompt per rep")
    for c in ("C1", "C2", "C1@32k"):
        d = r["decode"][c]
        check(len(d["rounds"]) == 2 and d["aggregate"]["median"] == statistics.median(d["aggregate"]["all"]), f"{c}: median of 2 rounds")
        for rd in d["rounds"]:
            s = rd["streams"]
            tot = sum(x["tokens"] for x in s)
            span = max(x["last"] for x in s) - min(x["first"] for x in s)
            check(rd["aggregate"] == round(tot / span, 2), f"{c}: aggregate = tokens / (last - first first)")
            per = statistics.mean((x["tokens"] - 1) / (x["last"] - x["first"]) for x in s)
            check(rd["per_stream_mean"] == round(per, 2), f"{c}: per-stream = (n - 1) / (last - first)")
            check(rd["finish"] == ["stop"], f"{c}: natural completion")
            check(all(40 <= x["tokens"] < 60 for x in s), f"{c}: tokens from usage.completion_tokens")
    check(r["decode"]["C2"]["serial"] is True, "C2 against a one-at-a-time server is flagged serial")
    check(r["decode"]["C1@32k"]["rounds"][0]["streams"][0]["prompt_tokens"] > 32768, "C1@32k carries the 32k prefix")
    dec = [b for b in MockServe.bodies if b["max_tokens"] != 1]
    pre = [b for b in MockServe.bodies if b["max_tokens"] == 1]
    check(len(pre) == 7 and len(dec) == 2 + 4 + 2, "1 warm-up + 6 prefill, 8 decode requests")
    check(all(b["temperature"] == 0 and b["stream"] and b["stream_options"]["include_usage"] for b in MockServe.bodies),
          "greedy, streamed, usage on")
    check(all(b.get("reasoning_budget_tokens") == 0 and b["max_tokens"] == SERVE_MAX_TOKENS for b in dec),
          "decode: thinking closed, serve's max_tokens cap")
    check(all("Write a detailed, well-structured explanation of" in b["messages"][0]["content"] for b in dec), "his question text")
    check(len(r["prompt_sha256"]) == 8 and len(set(r["prompt_sha256"][1:3])) == 2, "8 decode prompts, distinct within a round")
    # a serve that dies after the prefill: every decode cell is an error row, no ZeroDivisionError
    srv = http.server.HTTPServer(("127.0.0.1", 0), MockServe)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    MockServe.dead_decode = True
    out2 = os.path.join(tempfile.mkdtemp(), "sweep-dead.json")
    try:
        main(["--url", f"http://127.0.0.1:{srv.server_address[1]}", "--config", "self-test dead decode", "--out", out2,
              "--prefill", "256", "--conc", "1", "--reps", "1", "--dec-reps", "1", "--no-power"], words=words)
    finally:
        MockServe.dead_decode = False
        srv.shutdown()
    r2 = json.load(open(out2, encoding="utf-8"))
    check(r2["status"] == "DONE WITH ERRORS" and len(r2["errors"]) == 2, "dead decode: status and two errors")
    check(all("zero decode tokens" in r2["decode"][c]["error"] and "aggregate" not in r2["decode"][c] for c in ("C1", "C1@32k")),
          "dead decode: C1 and C1@32k are error rows without a rate")
    check(r2["prefill"]["256"]["n"] == 1, "dead decode: the prefill cell stands")
    real = DEFAULT_TOKENIZER if os.path.exists(DEFAULT_TOKENIZER) else None
    if real:
        w = vocab_words(real)
        check(len(w) > 10000 and all(x[0] == " " and x[1:].isalpha() for x in w), "tokenizer.json word list")
    print(f"self-test: {'FAIL ' + '; '.join(fails) if fails else 'ok'} ({out}; tokenizer.json word list "
          f"{'checked' if real else 'not found, skipped'})")
    return 1 if fails else 0


if __name__ == "__main__":
    if "--self-test" in sys.argv[1:]:
        sys.exit(self_test())
    sys.exit(1 if main().get("errors") else 0)
