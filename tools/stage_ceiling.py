#!/usr/bin/env python3
"""#240 step 5: the stage-ceiling arithmetic of a decode token, shared by tools/glm_tier_sim.py (`dyn_cost`,
`rates_from_file`) and tools/prequal.py. Stdlib only.

A token moves `bytes_per_token` through each stage at `rate_gbps` (GB/s, 10^9 B). A stage's ceiling is
rate * 1e9 / bytes tok/s (inf at 0 bytes, None without a rate). Under full overlap the lowest given ceiling
binds; the serial bound adds the stage times (no overlap) and exists only when every rate is given. These
are the semantics `glm_tier_sim.dyn_cost` had before #240 (glm_tier_sim.py:1020-1040 at 7b0fea8).

Rates come from a JSON file {"<key>": <GB/s>, ..., "source": "..."}: a present key must be a positive
number; a missing key leaves that stage without a rate, never a default.
"""
import json
import math
import os


def ceilings(stages):
    """stages = [(name, bytes_per_token, rate_gbps or None), ...] ->
    {name: {"bytes_per_token", "rate_gbps", "ceiling_tok_s"}, ..., "binding", "ceiling_tok_s", "serial_tok_s"}."""
    out, ceil, serial = {}, [], 0.0
    for name, by, rate in stages:
        c = None if rate is None else (math.inf if by == 0 else rate * 1e9 / by)
        out[name] = {"bytes_per_token": by, "rate_gbps": rate, "ceiling_tok_s": c}
        if c is not None:
            ceil.append((c, name))
        if serial is not None:
            serial = None if rate is None else serial + by / (rate * 1e9)
    out["binding"] = min(ceil)[1] if ceil else None
    out["ceiling_tok_s"] = min(ceil)[0] if ceil else None
    out["serial_tok_s"] = (math.inf if serial == 0 else 1.0 / serial) if serial is not None else None
    return out


def read_rates(path, keys, error=ValueError):
    """The rates of `keys` (GB/s) from a JSON file {"<key>": x, ..., "source": "..."} -> (rates, reason). A key
    may be missing (that stage then has no rate); a present value must be a positive number, else `error`.
    reason = "<basename> (<source>)"; no path -> ({}, "no rates file given"). A file that cannot be read or
    parsed raises OSError / ValueError (json.JSONDecodeError) as json.load does."""
    if not path:
        return {}, "no rates file given"
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    if not isinstance(d, dict):
        raise error("%s: not a JSON object {\"<key>\": GB/s, \"source\": ...}" % path)
    out = {}
    for key in keys:
        if key in d:
            x = d[key]
            if isinstance(x, bool) or not isinstance(x, (int, float)) or not x > 0:
                raise error("%s: %s = %r is not a positive number" % (path, key, x))
            out[key] = float(x)
    return out, "%s (%s)" % (os.path.basename(path), d.get("source", "no source named"))
