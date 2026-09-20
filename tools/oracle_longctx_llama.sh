#!/usr/bin/env bash
# #90: the LLAMA arm of the long-context oracle form - one command, whenever the
# GGUF is whole again (2026-09-20: shard 1 of the Unsloth UD-Q2_K_XL GGUF is
# broken/incomplete, so this is a harness, not a run).
#
# WHAT IT DOES when the server is back:
#   1. /props + the tokenizer round-trip check on the anchor's prefix ids
#      (the check that decides whether the ids survive as ids - docs/oracle-kld.md 4.1)
#   2. tools/llama-row-probs.py over the plan's row block, --no-cache-prompt
#      (every row a pure function of its prefix - the arm of record)
#   3. trim the dump to the plan's 64 rows (the sparse form oracle-kld.py reads)
#
# THE SAME CEILING THE ENGINE ARM HAS. llama-row-probs writes rows 0..last
# contiguously, so anchor 50000 would be a 50 GB file; anchors above 2564 are
# refused with that reason until it grows a write-window flag.
#
# Usage:  tools/oracle_longctx_llama.sh [--base-url http://127.0.0.1:8083]
#                                       [--anchors "1000 2564"]
#         (start the server first: ~/.local/share/crow/venv/bin/python \
#            ~/.local/share/crow/tools/start-server.py flash-next-q2-k-xl)

set -euo pipefail
cd "$(dirname "$0")/.."

BASE_URL=${BASE_URL:-http://127.0.0.1:8083}
ANCHORS="1000 2564"
VENV_PY=$HOME/.local/share/crow/venv/bin/python
while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-url) BASE_URL="$2"; shift 2 ;;
    --anchors)  ANCHORS="$2";  shift 2 ;;
    -h|--help)  sed -n '2,21p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown arg $1 (try --help)" >&2; exit 2 ;;
  esac
done

OUT=decode_out/oracle-longctx/llama
PLAN=decode_out/oracle-longctx/row-plan.json
PY=${VENV_PY:-python3}
[[ -x $PY ]] || PY=python3

if ! curl -fsS --max-time 5 "$BASE_URL/props" >/dev/null 2>&1; then
  echo "llama arm: no server at $BASE_URL - the GGUF shard 1 is broken as of" \
       "2026-09-20; start it with the command in this file's header and rerun" >&2
  exit 3
fi

for a in $ANCHORS; do
  (( a > 2564 )) && { echo "anchor $a: REFUSED - llama-row-probs writes rows 0..last \
contiguously ($(( a * 248320 * 4 / 1000000000 )) GB at this anchor); it needs a \
write-window flag for the deep anchors" >&2; continue; }
  ids=decode_out/oracle-longctx/longctx-170k-a${a}-ids.json
  dir=$OUT/a${a}
  mkdir -p "$dir"
  rows=$(python3 - "$PLAN" "$a" <<'EOF'
import json, sys
plan = json.load(open(sys.argv[1]))
g = [g for g in plan["groups"] if g["anchor"] == int(sys.argv[2])][0]
print("%d:%d" % (g["rows"][0], g["rows"][-1] + 1))
EOF
)
  first=${rows%%:*}; last=${rows##*:}
  echo "== anchor $a: round-trip check, then rows $rows"
  $PY tools/llama-row-probs.py --base-url "$BASE_URL" --ids "$ids" --round-trip \
      | tee "$dir/round-trip.log" || true
  $PY tools/llama-row-probs.py --base-url "$BASE_URL" --ids "$ids" --rows "0:$last" \
      --no-cache-prompt --out "$dir/gpu-logits.f32" 2>&1 | tee "$dir/collect.log"
  rowlist=$(python3 - "$PLAN" "$a" <<'EOF'
import json, sys
plan = json.load(open(sys.argv[1]))
g = [g for g in plan["groups"] if g["anchor"] == int(sys.argv[2])][0]
print(json.dumps(g["rows"]))
EOF
)
  python3 tools/oracle_longctx_rows.py subset --source "$dir/gpu-logits.f32" \
      --out "$dir/plan-rows.f32" --rows-file <(echo "$rowlist")
  sha256sum "$dir/gpu-logits.f32" "$dir/plan-rows.f32" "$ids" > "$dir/SHA256SUMS"
  rm -f "$dir/gpu-logits.f32"
  echo "anchor $a: plan rows in $dir/plan-rows.f32 (+ .rows.json)"
done
echo "llama arm done - read with tools/oracle-kld.py as an ordinary sparse --arm"
