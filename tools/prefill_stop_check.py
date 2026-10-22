#!/usr/bin/env python3
"""The prefill stop, live against a running `serve`: a client that leaves during a long prompt
ends the prefill at the next chunk boundary, and the next request resumes from there.

  python -I tools/prefill_stop_check.py --port 8199 [--doc docs/architecture.md] [--chars 48000]
                                        [--after 4.0] [--tokens 96]

Four requests, greedy, thinking off, one long user message (the first `--chars` characters of
`--doc` behind a fixed instruction):
  R  the long prompt, uninterrupted, `stream:false`: the reference answer
  X  an unrelated short prompt, so R's snapshot is no longer the held conversation
  A  the long prompt as a stream; `--after` s after the 200 head the client shuts its socket
     down in both directions and closes it, the way Crow's stop hangs up (`_hang_up`)
  B  the long prompt again, `stream:false`

Pass: B's `cached_tokens` is above 0 (it resumed from A's stop) and B's answer equals R's. The
engine side of the check is in serve's log: `[prefill] STOPPED at N/M tok` for A, then
`[cache] WARM ... P N` for B. Run against a serve with `CROW_PREFIX_PARK_ROWS=0`, or X parks R
and A restores it instead of prefilling. Exit 0 on pass, 1 on a fail, with one line per request.
Standard library only.
"""

import argparse
import json
import socket
import sys
import time
import urllib.request


def ask(url, messages, tokens):
    body = {"messages": messages, "max_tokens": tokens, "temperature": 0, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False}}
    started = time.monotonic()
    request = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=1800) as resp:
        doc = json.load(resp)
    return doc, time.monotonic() - started


def hang_up_after(port, messages, tokens, after):
    body = json.dumps({"messages": messages, "max_tokens": tokens, "temperature": 0,
                       "stream": True, "chat_template_kwargs": {"enable_thinking": False}}).encode("utf-8")
    sock = socket.create_connection(("127.0.0.1", port))
    sock.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                 b"Content-Type: application/json\r\nContent-Length: %d\r\n\r\n" % len(body) + body)
    head = sock.recv(4096).split(b"\r\n")[0].decode("ascii", "replace")
    time.sleep(after)
    sock.shutdown(socket.SHUT_RDWR)
    sock.close()
    return head


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--doc", default="docs/architecture.md")
    ap.add_argument("--chars", type=int, default=48000)
    ap.add_argument("--after", type=float, default=4.0)
    ap.add_argument("--tokens", type=int, default=96)
    a = ap.parse_args()
    url = "http://127.0.0.1:%d/v1/chat/completions" % a.port
    with open(a.doc, encoding="utf-8") as fh:
        text = fh.read()[:a.chars]
    long_prompt = [{"role": "user", "content":
                    "Read this, then summarize its first section in three sentences.\n\n" + text}]

    r, r_s = ask(url, long_prompt, a.tokens)
    print("R  %.1f s, prompt %d tok, cached %d, answer %d tok"
          % (r_s, r["usage"]["prompt_tokens"], r["usage"]["prompt_tokens_details"]["cached_tokens"],
             r["usage"]["completion_tokens"]))
    _, x_s = ask(url, [{"role": "user", "content": "Say hello, please."}], 8)
    print("X  %.1f s" % x_s)
    print("A  %s, hung up %.1f s after the head" % (hang_up_after(a.port, long_prompt, a.tokens, a.after), a.after))
    b, b_s = ask(url, long_prompt, a.tokens)
    cached = b["usage"]["prompt_tokens_details"]["cached_tokens"]
    print("B  %.1f s, prompt %d tok, cached %d, answer %d tok"
          % (b_s, b["usage"]["prompt_tokens"], cached, b["usage"]["completion_tokens"]))
    same = r["choices"][0]["message"] == b["choices"][0]["message"]
    print("answer: %s; resume: %s" % ("same as R" if same else "DIFFERENT from R",
                                     "from %d cached tokens" % cached if cached else "NONE (cold)"))
    return 0 if same and cached > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
