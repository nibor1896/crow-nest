import json, sys, time, urllib.request
url, out, maxt = sys.argv[1], sys.argv[2], int(sys.argv[3])
extra = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
b = {"messages": [{"role": "user", "content": "Write a detailed, well-structured explanation of how CPU caches work (variant 7)."}],
     "stream": True, "temperature": 0, "max_tokens": maxt, "stream_options": {"include_usage": True}, "crow_force_ids": []}
b.update(extra)
req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(b).encode(), headers={"Content-Type": "application/json"})
t0 = time.perf_counter(); times = []; text = []; reas = []; last = None
with urllib.request.urlopen(req, timeout=7200) as r:
    for raw in r:
        line = raw.decode("utf-8").strip()
        if not line.startswith("data:"): continue
        body = line[5:].strip()
        if body == "[DONE]": break
        last = json.loads(body); times.append(time.perf_counter() - t0)
        for ch in last.get("choices") or []:
            d = ch.get("delta") or {}
            text.append(d.get("content") or ""); reas.append(d.get("reasoning_content") or "")
json.dump({"times": times, "text": "".join(text), "reasoning": "".join(reas), "last": last}, open(out, "w"), indent=1)
print(out, "frames", len(times), "timings", (last or {}).get("timings"))
