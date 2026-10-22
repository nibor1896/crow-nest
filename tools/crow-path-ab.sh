#!/usr/bin/env bash
# Usage, from the repository root:
#   AB_ENV="CROW_STAGE_PAR=1" tools/crow-path-ab.sh <outdir> <order> <session.json> <K> [<K> ...]
#   <outdir>        logs and rows land here (one serve log per run, rows.jsonl, summary.txt)
#   <order>         one character per boot: O = serve exactly as Crow boots it, S = the same plus
#                   AB_ENV (space-separated NAME=VALUE pairs), e.g. OSOSOS
#   <session.json>  a SNAPSHOT of a Crow session (Crow rewrites the live file on exit)
#   <K>             replay points: messages[0:K] go out, the model answers message K
#   ROUNDS (4)      seeds per replay point; MAX_TOKENS (512) caps each answer; PORT (8199)
#   INSTALL (~/.local/share/crow), MODELS (INSTALL/models): where Crow's install lives
#
# WHAT THIS ANSWERS: whether a serve switch changes speed on Crow's real request path, which
# is where a speed lever has to be measured before a default changes (2026-10-06: a greedy,
# tool-free A/B read CROW_STAGE_PAR as +6.7 % on Windows, Crow's own session then halved).
#   - the engine is booted the way Crow boots the flash-next point: Crow's installed binary,
#     cwd and env from manifests/stack.json (points[flash-next].engine), port PORT
#   - the requests are Crow's own: tools/corruption-replay-probe.py builds every body through
#     crow_core.stream_reply (tools, tool grammar, Crow's sampling row, streaming), seeds
#     0..ROUNDS-1, so both arms draw the same answers if the switch is bit-identical
#   - one fresh process per boot, arms alternating as given, nothing discarded
# WHAT EVERY BOOT RECORDS: its serve log, and per request the serve `routing` JSON line
# (prompt_n, cached_n, predicted_n, prompt_ms, predicted_ms, tok_s, cold, ...) as a row of
# rows.jsonl with the arm and the boot index. summary.txt: per arm, the median decode tok/s of
# the requests that generated at least 64 tokens, per boot and over all boots, and the median
# prefill tok/s of the requests that prefilled at least 256 tokens.
# MACHINE RULES: one engine at a time. Before every boot no `serve` may run and the GPU must be
# under 900 MiB; every serve is stopped by pid and the next boot waits for it to be gone.
set -u

[ $# -ge 4 ] || { sed -n '2,13p' "$0"; exit 2; }
out="$1"; order="$2"; session="$3"; shift 3; points=("$@")
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
install="${INSTALL:-$HOME/.local/share/crow}"
models="${MODELS:-$install/models}"
port="${PORT:-8199}"; rounds="${ROUNDS:-4}"; max_tokens="${MAX_TOKENS:-512}"
mkdir -p "$out"
cp "$session" "$out/session.json"

# Crow's env for the flash-next point, ${INSTALL} and ${MODELS} expanded
mapfile -t crow_env < <(python3 -I - "$install/manifests/stack.json" "$install" "$models" <<'EOF'
import json, sys
doc = json.load(open(sys.argv[1], encoding="utf-8"))
point = next(p for p in doc["points"] if p.get("id") == "flash-next")
for k, v in point["engine"]["env"].items():
    print("%s=%s" % (k, v.replace("${INSTALL}", sys.argv[2]).replace("${MODELS}", sys.argv[3])))
EOF
)
[ ${#crow_env[@]} -gt 0 ] || { echo "crow-path-ab: no flash-next env in $install/manifests/stack.json" >&2; exit 2; }

gpu_mib() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }
ats=(); for k in "${points[@]}"; do ats+=(--at "$k"); done

: > "$out/rows.jsonl"
for ((i = 0; i < ${#order}; i++)); do
    arm="${order:i:1}"
    case "$arm" in O) extra=() ;; S) read -r -a extra <<< "${AB_ENV:-}" ;; *) echo "crow-path-ab: arm '$arm' is not O or S" >&2; exit 2 ;; esac
    if pgrep -x serve >/dev/null; then echo "crow-path-ab: a serve is running - stop it first" >&2; exit 2; fi
    for _ in $(seq 1 60); do [ "$(gpu_mib)" -lt 900 ] && break; sleep 1; done
    log="$out/serve-$i-$arm.log"
    ( cd "$install" && exec env "${crow_env[@]}" "${extra[@]}" LD_LIBRARY_PATH="$install/bin" \
        "$install/bin/serve" --port "$port" ) > "$log" 2>&1 &
    pid=$!
    up=0
    for _ in $(seq 1 120); do
        curl -s -m 2 "http://127.0.0.1:$port/health" | grep -q '"ok"' && { up=1; break; }
        kill -0 "$pid" 2>/dev/null || break
        sleep 2
    done
    if [ $up -eq 1 ]; then
        python3 -I "$root/tools/corruption-replay-probe.py" --session "$out/session.json" "${ats[@]}" \
            --rounds "$rounds" --port "$port" --served-name auto --label "$arm" \
            --sampling "{\"max_tokens\": $max_tokens}" --json "$out/probe-$i-$arm.json" > "$out/probe-$i-$arm.txt" 2>&1
        echo "boot $i arm $arm: probe exit $?"
    else
        echo "boot $i arm $arm: serve did not come up, see $log"
    fi
    kill -TERM "$pid" 2>/dev/null
    for _ in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    grep -o '{"ts".*"event":"routing".*' "$log" | python3 -I -c '
import json, sys
for line in sys.stdin:
    row = json.loads(line); row["arm"] = sys.argv[1]; row["boot"] = int(sys.argv[2]); print(json.dumps(row))
' "$arm" "$i" >> "$out/rows.jsonl"
done

python3 -I - "$out/rows.jsonl" "${AB_ENV:-}" <<'EOF' | tee "$out/summary.txt"
import json, statistics, sys
rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
print("S = O + %s; decode: requests with >= 64 generated tokens; prefill: >= 256 prefilled" % sys.argv[2])
for arm in "OS":
    mine = [r for r in rows if r["arm"] == arm]
    dec = [r for r in mine if r["predicted_n"] >= 64]
    pre = [r for r in mine if r["prompt_n"] - r["cached_n"] >= 256]
    boots = sorted({r["boot"] for r in mine})
    per_boot = ["%.2f" % statistics.median(r["tok_s"] for r in dec if r["boot"] == b)
                for b in boots if any(r["boot"] == b for r in dec)]
    print("%s  boots %s, requests %d: decode median %s tok/s (per boot %s), prefill median %s tok/s (n %d)" % (
        arm, boots, len(mine),
        "%.2f" % statistics.median(r["tok_s"] for r in dec) if dec else "-", ", ".join(per_boot),
        "%.1f" % statistics.median((r["prompt_n"] - r["cached_n"]) / r["prompt_ms"] * 1e3 for r in pre) if pre else "-",
        len(pre)))
EOF
