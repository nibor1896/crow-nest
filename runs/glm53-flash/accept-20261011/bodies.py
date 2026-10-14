"""Request bodies for the 2026-10-11 GPU acceptance block (#215 #219 #217 #216, combined).

Rebuilt the way Crow v3.2.9 sends them (Crow/cli/crow_core.py `stream_reply`: model, messages =
Conversation.payload(), temperature 1.0, top_p 0.95, min_p 0.01, stream, stream_options,
timings_per_token, tools = TOOLS (31), max_tokens 16384, seed), from Crow's stored G5 chats:
  t1 = %LOCALAPPDATA%/crow/session/chat-20261011-124133.json
  t2 = %LOCALAPPDATA%/crow/session/session.json
Selftest: `python -I bodies.py` prints each body's byte size against the G5 serve log.
"""
import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CROW = r"<home>\dev\Crow\cli"
SESS = os.path.join(os.environ["LOCALAPPDATA"], "crow", "session")
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
MODEL_DIR = os.path.join(REPO, "models", "GLM-5.3-Flash-original")

T1_FILE = os.path.join(SESS, "chat-20261011-124133.json")
T2_FILE = os.path.join(SESS, "session.json")
T1_SEEDS = (1408639702, 559453182, 772167202)
T2_SEED1 = 1145815877


def tools():
    cache = os.path.join(HERE, "crow_tools.json")
    if not os.path.exists(cache):
        sys.path.insert(0, CROW)
        import crow_core  # noqa: E402
        json.dump(crow_core.TOOLS, open(cache, "w", encoding="utf-8"))
    return json.load(open(cache, encoding="utf-8"))


def msgs(path):
    return json.load(open(path, encoding="utf-8"))["messages"]


def body(messages, seed, max_tokens=16384, **extra):
    b = {"model": "crow", "messages": [dict(m) for m in messages], "temperature": 1.0, "top_p": 0.95,
         "min_p": 0.01, "stream": True, "stream_options": {"include_usage": True}, "timings_per_token": True,
         "tools": tools(), "max_tokens": max_tokens}
    b["seed"] = seed
    b.update(extra)
    return b


def t1(i, **kw):
    """t1 request i (1..3): messages 0-1, 0-4, 0-6."""
    m = msgs(T1_FILE)
    n = {1: 2, 2: 5, 3: 7}[i]
    return body(m[:n], T1_SEEDS[i - 1], **kw)


def t2(i, **kw):
    """t2 request i: 1 = messages 0-1, 2 = 0-3, ... (G5 rows 4..9)."""
    m = msgs(T2_FILE)
    n = {1: 2, 2: 4, 3: 8, 4: 12, 5: 16, 6: 18}[i]
    return body(m[:n], kw.pop("seed", T2_SEED1 if i == 1 else 0), **kw)


THIRD_AREA = r"<home>\Desktop\g5-run1\t3-accept"


def third(**kw):
    """The third new chat of #217's amendment: t2's system message with the working area changed to
    THIRD_AREA, t1's user message (another task), seed 1408639702."""
    m = copy.deepcopy(msgs(T2_FILE)[:2])
    t1m = msgs(T1_FILE)
    old = json.load(open(T2_FILE, encoding="utf-8"))["crow_root"]
    assert old in m[0]["content"], "t2's working area not found in its system message"
    m[0]["content"] = m[0]["content"].replace(old, THIRD_AREA)
    m[1] = dict(t1m[1])
    return body(m, kw.pop("seed", 1408639702), **kw)


if __name__ == "__main__":
    for name, b, want in [("t1r1", t1(1), 27159), ("t1r2", t1(2), 36479), ("t1r3", t1(3), 36867)]:
        print(name, len(json.dumps(b)), "G5 log", want)
    print("t2r1", len(json.dumps(t2(1))), "third", len(json.dumps(third())))
