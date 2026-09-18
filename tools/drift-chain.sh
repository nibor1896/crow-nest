#!/usr/bin/env bash
# Usage, from anywhere:  tools/drift-chain.sh <label> <order> [outdir]
#   <label>   names the chain; the logs land in decode_out/38/<label> (gitignored)
#   <order>   one character per run: S = serve arm, D = `decode run` arm, e.g. SDSDSDSD
#   [outdir]  overrides the log directory
#
# WHAT THIS ANSWERS (issue #38). On Windows, 2026-09-10, two serve runs of the SAME request in
# one chain read 30.45 and 22.42 tok/s over 255 timed decode steps - bit-identical generated ids,
# identical engine counters, 26 % apart (`decode_out/srv-m2a.log:824`). Four later chains
# (#37, 2026-09-11) spread only 1.004 to 1.105 and did not reproduce it. The issue's own first
# step is one chain that runs the serve arm several times back to back, one fresh process per run,
# and records enough per run to say whether the drift is MONOTONIC in run index, a FIRST-RUN
# effect, or absent. This script is that chain on Linux.
#
# THE TWO ARMS, alternating as adjacent pairs the way the #37 chains ran them:
#   S  `serve` (tools/serve-linux.sh, port 8099) answering ONE POST /v1/chat/completions
#      with the t1-read prompt as a single user message, `max_tokens` 256, `temperature` 0
#      (greedy), `stream` false. serve renders the chat template itself, so the prompt it
#      prefills is the 16,064 ids of record - verified per run against the id count on the
#      `[chat] prompt` line, and the request body is built from `decode_out/ten-tasks.json`.
#   D  `decode run decode_out/srv-a5-t1read-ids.json 256` - the same 16,064 ids, fed as ids,
#      the arm the #37 chains called D1.
# Both arms generate 256 tokens and both time 255 steps: serve's `tok_s` is
# (generated - 1) / decode_ms (`bin/serve.rs:3102`) and `decode run`'s mean is over `gen - 1`
# latencies (`bin/decode.rs:266`). NO warmup run is discarded: the question IS whether run 1
# differs from run 4, so discarding it would delete the measurement.
#
# WHAT EVERY RUN RECORDS
#   before the engine starts (machine.jsonl, one line per run): UTC, free-for-pinning computed
#   with the engine's own formula (MemTotal - (AnonPages + Shmem + SUnreclaim + KernelStack +
#   PageTables + Percpu), architecture 8.8 point 2), MemAvailable, `Cached` (the page cache),
#   and nvidia-smi memory.used / clocks.sm / power.draw / temperature.gpu;
#   after the run (rows.jsonl, one line per run): the `[budget]` boot line's own budget and
#   free-for-pin, the boot JSON line, for S the `routing` JSON line (predicted_ms, tok_s,
#   selections, cold, bytes_streamed, ple_rows, ple_fills, trickle_swaps) and for D the
#   `decode:` / `cold experts per timed decode token` / `ple rows per timed decode token` lines
#   plus `decode_out/run.json`, and the sha256 of the generated ids.
# The ids sha256 is the STOP condition: within one arm every run must carry the same sha
# (`CROW_LOG=info,chat=debug` is what brings serve's `[chat] ids` list back since #13). The two
# arms are offset by one token by construction - `decode run` discards a warm-up step whose
# token it keeps generating from - so the cross-arm check is serve ids[1:] against decode
# trace[:255], printed as `cross_arm_ids`.
#
# MACHINE RULES, the same ones every chain in this repository runs under
#   One engine at a time: before every start `pgrep` must find no serve/decode/parity, the GPU
#   must be back under 900 MiB, and `engine/.engine.lock` must be absent - a lock whose pid is
#   dead is removed and the removal is logged. Every serve is stopped BY PID and the chain waits
#   for the process to be gone and the GPU to fall back before the next load.
#   Both arms run inside the same transient scope as tools/serve-linux.sh
#   (`systemd-run --user --scope --slice=session.slice`, MemorySwapMax=0,
#   MemoryHigh=MemTotal-8G, MemoryMax=MemTotal-6G, computed from /proc/meminfo) - the D arm
#   deliberately takes the launcher's computed limits instead of the hard-coded 52G/54G of
#   tools/gate-linux.sh, so that the two arms of one chain are bounded identically.
#   The Windows "> 50.5 GiB free" gate of #38 is NOT this script's gate: on Linux the derived
#   pinned budget of v0.3.0 replaced it (`[budget]` line, architecture 8.8), so the chain
#   RECORDS free-for-pin per run instead of gating on a hand-picked number.
# Environment: CUDA_LIB (default ~/.local/share/crow/cuda/lib, never lib/stubs), CROW_CNQ,
#   CROW_HOTSETS, DRIFT_PORT (not a CROW_ name on purpose: tools/serve-linux.sh passes every
#   CROW_* of the caller through to the engine). Nothing else of the caller reaches the engine.

set -u

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
label="${1:-}"
order="${2:-SDSDSDSD}"
[ -n "$label" ] || { echo "usage: tools/drift-chain.sh <label> <order SDSD...> [outdir]" >&2; exit 2; }
out="${3:-$root/decode_out/38/$label}"
case "$out" in /*) ;; *) out="$root/$out" ;; esac

PORT="${DRIFT_PORT:-8099}"
GEN=256                       # 256 generated tokens => 255 timed decode steps in both arms
GPU_IDLE_MIB=900              # the desktop alone reads 650-700 MiB on this box
IDS="$root/decode_out/srv-a5-t1read-ids.json"
TASKS="$root/decode_out/ten-tasks.json"
CNQ="${CROW_CNQ:-converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq}"
HOTSETS="${CROW_HOTSETS:-decode_out/hotsets-M-longctx2100-n160.json}"
cuda_lib="${CUDA_LIB:-$HOME/.local/share/crow/cuda/lib}"
decode_bin="$root/engine/target/release/decode"
lock="$root/engine/.engine.lock"

for f in "$decode_bin" "$root/engine/target/release/serve" "$IDS" "$TASKS" "$root/tools/serve-linux.sh"; do
    [ -e "$f" ] || { echo "drift-chain.sh: missing $f" >&2; exit 2; }
done
[ -d "$cuda_lib" ] || { echo "drift-chain.sh: no CUDA runtime directory at $cuda_lib - set CUDA_LIB" >&2; exit 2; }

mem_total_kb=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)
high=$(( mem_total_kb * 1024 - 8 * 1024 * 1024 * 1024 ))
max=$(( mem_total_kb * 1024 - 6 * 1024 * 1024 * 1024 ))

mkdir -p "$out"
cd "$root"
chain_log="$out/chain.log"
rows="$out/rows.jsonl"
machines="$out/machine.jsonl"
: > "$rows"; : > "$machines"

say() { printf '%s\n' "$*" | tee -a "$chain_log"; }

body="$out/request.json"
python3 - "$TASKS" "$body" "$GEN" <<'PY'
import json, sys
tasks = json.load(open(sys.argv[1]))
t = [x for x in tasks if x["id"] == "t1-read"][0]
json.dump({"messages": [{"role": "user", "content": t["text"]}],
           "max_tokens": int(sys.argv[3]), "temperature": 0, "stream": False,
           "timings_per_token": True}, open(sys.argv[2], "w"))
print("request body: t1-read, %d chars, max_tokens %s, temperature 0" % (len(t["text"]), sys.argv[3]))
PY
[ -s "$body" ] || { echo "drift-chain.sh: could not build the request body" >&2; exit 2; }

mi() { awk -v k="$1:" '$1 == k {print $2}' /proc/meminfo; }

# the machine block, before every engine start
machine_block() {
    local i="$1" arm="$2" utc ffp memav cached gpu used sm pw tp
    utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    ffp=$(awk -v a="$(mi MemTotal)" -v b="$(mi AnonPages)" -v c="$(mi Shmem)" -v d="$(mi SUnreclaim)" \
              -v e="$(mi KernelStack)" -v f="$(mi PageTables)" -v g="$(mi Percpu)" \
              'BEGIN{printf "%.2f", (a-(b+c+d+e+f+g))/1048576}')
    memav=$(awk -v c="$(mi MemAvailable)" 'BEGIN{printf "%.2f", c/1048576}')
    cached=$(awk -v c="$(mi Cached)" 'BEGIN{printf "%.2f", c/1048576}')
    gpu=$(nvidia-smi --query-gpu=memory.used,clocks.sm,power.draw,temperature.gpu --format=csv,noheader,nounits)
    used=$(echo "$gpu" | cut -d, -f1 | tr -d ' ')
    sm=$(echo "$gpu" | cut -d, -f2 | tr -d ' ')
    pw=$(echo "$gpu" | cut -d, -f3 | tr -d ' ')
    tp=$(echo "$gpu" | cut -d, -f4 | tr -d ' ')
    printf '{"run":%d,"arm":"%s","utc":"%s","free_for_pin_gib":%s,"mem_available_gib":%s,"cached_gib":%s,"gpu_used_mib":%s,"sm_mhz":%s,"power_w":%s,"temp_c":%s}\n' \
        "$i" "$arm" "$utc" "$ffp" "$memav" "$cached" "$used" "$sm" "$pw" "$tp" >> "$machines"
    say "  machine r$i $arm  $utc  free-for-pin ${ffp} GiB  MemAvailable ${memav} GiB  Cached ${cached} GiB  gpu ${used} MiB  sm ${sm} MHz  ${pw} W  ${tp} C"
}

# one engine at a time, no lock, GPU back to idle
quiet() {
    local procs used pid
    procs=$(pgrep -af 'target/release/(serve|decode|parity)' | tr '\n' ' ')
    [ -z "$procs" ] || { say "  precheck: an engine is alive ($procs)"; return 1; }
    procs=$(pgrep -a -x 'serve|decode|parity' | tr '\n' ' ')
    [ -z "$procs" ] || { say "  precheck: an engine is alive by name ($procs)"; return 1; }
    if [ -f "$lock" ]; then
        pid=$(tr -dc '0-9' < "$lock")
        if [ -n "$pid" ] && [ -d "/proc/$pid" ]; then say "  precheck: .engine.lock held by LIVE pid $pid"; return 1; fi
        say "  precheck: .engine.lock is stale (pid ${pid:-none} is dead) - removed"
        rm -f "$lock"
    fi
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    [ "${used:-9999}" -le "$GPU_IDLE_MIB" ] || { say "  precheck: GPU still holds ${used} MiB"; return 1; }
    return 0
}
wait_quiet() {
    local k=0
    until quiet; do k=$((k+1)); [ "$k" -ge 90 ] && { say "  precheck: NOT quiet after 180 s - stopping the chain"; return 1; }; sleep 2; done
    return 0
}

run_serve() {
    local i="$1" log="$out/r$i-serve.log" resp="$out/r$i-resp.json" wrapper spid k code t0 t1
    CROW_LOG=info,chat=debug CROW_CNQ="$CNQ" CROW_HOTSETS="$HOTSETS" \
        nohup "$root/tools/serve-linux.sh" --port "$PORT" > "$log" 2>&1 &
    wrapper=$!
    spid=""; k=0
    while [ -z "$spid" ] && [ "$k" -lt 60 ]; do spid=$(pgrep -x serve | head -1); [ -z "$spid" ] && sleep 1; k=$((k+1)); done
    [ -n "$spid" ] || { say "  r$i S: serve never appeared"; return 1; }
    say "  r$i S: serve pid $spid, loading"
    k=0
    until curl -s -m 5 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ok"'; do
        [ -d "/proc/$spid" ] || { say "  r$i S: serve died during the load, see $log"; return 1; }
        k=$((k+1)); [ "$k" -ge 300 ] && { say "  r$i S: no /health after 300 s"; kill "$spid" 2>/dev/null; return 1; }
        sleep 1
    done
    say "  r$i S: loaded after ${k} s, sending the request"
    t0=$(date +%s%3N)
    code=$(curl -s -m 900 -o "$resp" -w '%{http_code}' -H 'Content-Type: application/json' \
        --data-binary @"$body" "http://127.0.0.1:$PORT/v1/chat/completions")
    t1=$(date +%s%3N)
    say "  r$i S: http $code, wall $(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.1f", (b-a)/1000}') s"
    kill "$spid" 2>/dev/null
    k=0
    while [ -d "/proc/$spid" ]; do k=$((k+1)); [ "$k" -ge 120 ] && { say "  r$i S: pid $spid still alive after 120 s"; break; }; sleep 1; done
    wait "$wrapper" 2>/dev/null
    say "  r$i S: pid $spid gone after ${k} s"
    [ "$code" = "200" ] || return 1
    return 0
}

run_decode() {
    local i="$1" log="$out/r$i-decode.log" rc
    systemd-run --user --scope --slice=session.slice --quiet \
        -p MemorySwapMax=0 -p "MemoryHigh=$high" -p "MemoryMax=$max" \
        env CROW_CNQ="$CNQ" CROW_HOTSETS="$HOTSETS" CROW_GRAPH=1 CROW_MMA=1 CROW_LOG=info \
            "LD_LIBRARY_PATH=$cuda_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
            "$decode_bin" run "$IDS" "$GEN" "$out/r$i-d" > "$log" 2>&1
    rc=$?
    cp -f "$root/decode_out/run.json" "$out/r$i-run.json" 2>/dev/null
    [ "$rc" -eq 0 ] || { say "  r$i D: decode exit $rc, see $log"; return 1; }
    say "  r$i D: done"
    return 0
}

# one JSONL row per run, parsed from that run's log (and run.json for the D arm)
extract() {
    local i="$1" arm="$2"
    python3 - "$i" "$arm" "$out" "$GEN" "$rows" <<'PY'
import hashlib, json, re, sys
i, arm, out, gen, rows = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
log = f"{out}/r{i}-" + ("serve.log" if arm == "S" else "decode.log")
txt = open(log, errors="replace").read().splitlines()
row = {"run": int(i), "arm": arm, "log": log}

def ids_sha(ids):
    return hashlib.sha256(",".join(str(x) for x in ids).encode()).hexdigest()

for ln in txt:
    s = ln.strip()
    if s.startswith("{"):
        try:
            j = json.loads(s)
        except Exception:
            continue
        if j.get("event") == "operating_point":
            row["boot"] = {k: j[k] for k in ("n_ctx", "prompt_chunk", "residency_n", "kernel_path") if k in j}
            cp = j.get("cold_path", {})
            row["boot"]["pinned_bytes"] = cp.get("pinned_bytes")
            row["boot"]["expert_bytes"] = cp.get("expert_bytes")
        elif str(j.get("route", "")).startswith("chat"):
            row["routing"] = j
    elif "[budget] host pinned budget" in s:
        m = re.search(r"budget ([\d.]+) GiB \(([^)]*)\); free for pinning ([\d.]+) GiB", s)
        if m and "budget_gib" not in row:
            row["budget_gib"], row["budget_basis"], row["budget_free_for_pin_gib"] = float(m[1]), m[2], float(m[3])

if arm == "S":
    ids = None
    for ln in txt:
        m = re.search(r"\[chat\] ids \[([\d, ]*)\]", ln)
        if m:
            ids = [int(x) for x in m[1].split(",") if x.strip()]
    for ln in txt:
        m = re.search(r"\[chat\] prompt (\d+) tok .* generated (\d+) tok, prefill ([\d.]+) ms \(([\d.]+) tok/s\), reset ([\d.]+) ms, decode ([\d.]+) ms, ([\d.]+) tok/s, finish (\w+)", ln)
        if m:
            row.update(prompt_n=int(m[1]), generated=int(m[2]), prompt_ms=float(m[3]),
                       prefill_tok_s=float(m[4]), reset_ms=float(m[5]),
                       predicted_ms=float(m[6]), tok_s=float(m[7]), finish=m[8])
    if ids is not None:
        row["ids_n"], row["ids_sha256"] = len(ids), ids_sha(ids)
        json.dump(ids, open(f"{out}/r{i}-ids.json", "w"))
    r = row.get("routing", {})
    if r:
        row["ple_fills"], row["ple_rows"] = r["ple_fills"], r["ple_rows"]
        row["trickle_swaps"] = r["trickle_swaps"]
        # the `[chat]` line rounds the rate to one decimal; the routing line carries the
        # value the spread is computed from, and a 26 % question deserves the full precision
        row["tok_s_chat_line"], row["predicted_ms_chat_line"] = row.get("tok_s"), row.get("predicted_ms")
        row["tok_s"], row["predicted_ms"] = r["tok_s"], r["predicted_ms"]
        row["cold_request"], row["sel_request"] = r["cold"], r["selections"]
        row["bytes_request"] = r["bytes_streamed"]
        row["expert_bytes"] = round(r["bytes_streamed"] / max(r["cold"], 1))
else:
    try:
        rj = json.load(open(f"{out}/r{i}-run.json"))
    except Exception:
        rj = {}
    if rj:
        row.update(prompt_n=rj["prompt"], generated=rj["generated"], mean_ms=rj["mean_ms"],
                   p50_ms=rj["p50_ms"], tok_s=rj["tok_s"], prefill_s=rj["prefill_s"],
                   prefill_tok_s=rj["prefill_tok_s"], context=rj["context"], n_hot=rj["n_hot"])
        row["predicted_ms"] = rj["mean_ms"] * (gen - 1)
        row["ids_n"], row["ids_sha256"] = len(rj["trace"]), ids_sha(rj["trace"])
        json.dump(rj["trace"], open(f"{out}/r{i}-ids.json", "w"))
    for ln in txt:
        m = re.search(r"cold experts per timed decode token: ([\d.]+) of (\d+) selections -> (\d+) MB/token", ln)
        if m:
            row["cold_per_tok"], row["sel_per_tok"], row["mb_per_tok"] = float(m[1]), float(m[2]), float(m[3])
        m = re.search(r"ple rows per timed decode token: ([\d.]+) requested, ([\d.]+) misses", ln)
        if m:
            row["ple_rows_per_tok"], row["ple_fills_per_tok"] = float(m[1]), float(m[2])
        # the prefill+warm-up half of the same counters, so the D arm can be put on the
        # S arm's footing: serve's `routing` line is the REQUEST total (prefill included).
        m = re.search(r"cold experts during prefill\+warm-up: ([\d.]+) per token of ([\d.]+) selections \((\d+) tokens\)", ln)
        if m:
            row["cold_prefill_per_tok"], row["prefill_tokens"] = float(m[1]), int(m[3])
        m = re.search(r"decode: mean ([\d.]+) ms  p50 ([\d.]+) ms  \(([\d.]+) tok/s\)", ln)
        if m:
            row.setdefault("mean_ms", float(m[1]))
            row.setdefault("tok_s", float(m[3]))

# the D arm baselines its counters AFTER prefill and the warm-up step, serve drains the block
# once per REQUEST - so the one figure both arms can carry is cold selections per token
# PROCESSED (prompt + generated), which is how the #37 comment normalised its serve arm.
if arm == "D" and row.get("cold_per_tok") is not None:
    # the boot line's own expert_bytes (gu + dn of one expert), not the rounded MB/token print
    row["expert_bytes"] = row.get("boot", {}).get("expert_bytes") or \
        round(row["mb_per_tok"] * 1e6 / max(row["cold_per_tok"], 1e-9))
    row["cold_request"] = round(row["cold_per_tok"] * (gen - 1)
                                + row.get("cold_prefill_per_tok", 0.0) * row.get("prefill_tokens", 0))
    row["bytes_request"] = round(row["cold_request"] * row["expert_bytes"])
if row.get("cold_request") and row.get("prompt_n"):
    tok = row["prompt_n"] + row.get("generated", gen)
    row["cold_per_tok_all"] = row["cold_request"] / tok
    row["mb_per_tok_all"] = row.get("bytes_request", 0) / tok / 1e6

with open(rows, "a") as f:
    f.write(json.dumps(row) + "\n")
def f2(x):
    return "-" if x is None else f"{x:.2f}"
print("  r%s %s: tok/s %s  predicted_ms %s  cold/tok all %s  cold/tok decode %s  request cold %s  ids %s %s" % (
    i, arm, f2(row.get("tok_s")), f2(row.get("predicted_ms")), f2(row.get("cold_per_tok_all")),
    f2(row.get("cold_per_tok")), row.get("cold_request"), row.get("ids_n"),
    str(row.get("ids_sha256"))[:12]))
PY
}

say "== drift-chain.sh  $label  order $order  $(date -Is)  $(git -C "$root" rev-parse --short HEAD 2>/dev/null)  -> $out"
say "   prompt t1-read 16,064 ids (serve renders it, decode reads $IDS), $GEN tokens, $((GEN-1)) timed steps, greedy"
say "   scope MemoryHigh=$high MemoryMax=$max MemorySwapMax=0 on both arms"

n=${#order}
i=0
failed=0
while [ "$i" -lt "$n" ]; do
    i=$((i+1))
    arm="${order:$((i-1)):1}"
    say "-- run $i of $n, arm $arm"
    wait_quiet || { failed=1; break; }
    machine_block "$i" "$arm"
    if [ "$arm" = "S" ]; then run_serve "$i" || failed=1; else run_decode "$i" || failed=1; fi
    extract "$i" "$arm" | tee -a "$chain_log"
    [ "$failed" -eq 0 ] || { say "  run $i failed - stopping the chain"; break; }
done

# the last serve of a chain leaves its lock behind (no SIGTERM handler, machine rules): the
# next run's precheck would remove it, and there is no next run, so the chain does it here
if [ -f "$lock" ]; then
    lpid=$(tr -dc '0-9' < "$lock")
    if [ -n "$lpid" ] && [ -d "/proc/$lpid" ]; then say "  .engine.lock still held by LIVE pid $lpid - left in place"
    else say "  .engine.lock of the last run (pid ${lpid:-none}, dead) removed"; rm -f "$lock"; fi
fi

# the table, the within-arm spreads (max over min, the #37 comment's form) and the ids gate
python3 - "$rows" "$label" "$GEN" <<'PY' | tee -a "$chain_log"
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
label, gen = sys.argv[2], int(sys.argv[3])
print(f"\n== {label}: the chain table")
print("| run | arm | tok/s | predicted_ms | cold/tok decode | cold/tok all | request cold total | bytes streamed | ple fills | free-for-pin GiB | budget GiB | sm MHz | ids sha256 |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
mach = {}
try:
    for l in open(sys.argv[1].replace("rows.jsonl", "machine.jsonl")):
        m = json.loads(l); mach[m["run"]] = m
except FileNotFoundError:
    pass
for r in rows:
    m = mach.get(r["run"], {})
    bs = r.get("bytes_request")
    fills = r.get("ple_fills")
    if fills is None and r.get("ple_fills_per_tok") is not None:
        fills = round(r["ple_fills_per_tok"] * (gen - 1))
    print("| %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
        r["run"], r["arm"],
        f"{r['tok_s']:.2f}" if r.get("tok_s") else "-",
        f"{r['predicted_ms']:.1f}" if r.get("predicted_ms") else "-",
        f"{r['cold_per_tok']:.1f}" if r.get("cold_per_tok") else "-",
        f"{r['cold_per_tok_all']:.1f}" if r.get("cold_per_tok_all") else "-",
        f"{r['cold_request']:,}" if r.get("cold_request") else "-",
        f"{bs:,}" if bs else "-", f"{fills:,}" if fills else "-",
        m.get("free_for_pin_gib", "-"), r.get("budget_gib", "-"), m.get("sm_mhz", "-"),
        str(r.get("ids_sha256"))[:12]))
print("\n`bytes streamed` is the routing line's own definition, cold selections x expert bytes: it")
print("over-counts prefill, where one chunk stages an expert once for many tokens. Within one arm")
print("every run does the identical work, which is what the spread column is read against.")
print("`cold/tok decode` is decode-only and exists for arm D alone: serve drains the device counter")
print("block once per REQUEST, so its count covers prefill + the 255 decode steps. `cold/tok all`")
print("and `request cold total` are the figures both arms carry, per token PROCESSED.")
print("\n== within-arm spread, max over min (the #37 comment's form)")
print("| arm | runs | positions | min tok/s | max tok/s | mean tok/s | spread |")
print("|---|---|---|---|---|---|---|")
for arm in ("S", "D"):
    a = [r for r in rows if r["arm"] == arm and r.get("tok_s")]
    if not a:
        continue
    v = [r["tok_s"] for r in a]
    print("| %s | %d | %s | %.2f | %.2f | %.2f | %.3f |" % (
        arm, len(v), ", ".join(str(r["run"]) for r in a), min(v), max(v), sum(v)/len(v), max(v)/min(v)))
    shas = {r.get("ids_sha256") for r in a}
    print(f"|  ids sha256 in arm {arm} | {len(shas)} distinct | {'IDENTICAL' if len(shas)==1 else 'DIFFER - STOP'} | | | | |")
s = [r for r in rows if r["arm"] == "S" and r.get("ids_sha256")]
d = [r for r in rows if r["arm"] == "D" and r.get("ids_sha256")]
if s and d:
    import hashlib, os
    out = os.path.dirname(sys.argv[1])
    si = json.load(open(f"{out}/r{s[0]['run']}-ids.json"))
    di = json.load(open(f"{out}/r{d[0]['run']}-ids.json"))
    print("\ncross_arm_ids: serve[1:] == decode[:%d] -> %s" % (len(si) - 1, si[1:] == di[:len(si) - 1]))
PY

say "== drift-chain.sh: $([ "$failed" -eq 0 ] && echo 'chain complete' || echo 'chain STOPPED - see above')  $(date -Is)"
exit "$failed"
