#!/bin/bash
# 3-minute check, robin 2026-10-10: one cold 8192-token prefill of real text (held-out todo-1006, first 8192 ids)
# + 128 cold decode tokens on a real question (no random ids: they fall into a two-token loop).
# Usage: quick.sh <label> [extra env...]   -> runs/glm53-flash/quick/<label>/
# The operating set of the evening of 2026-10-10 (b6-promptcount) is arm3.env = arm2.env + RT2 + 13 switches:
#   quick.sh <label> $(cat arm3.env)   (README.md in this directory lists every arm and its figures)
W=/c/Users/robin/dev/crow-nest-wt-int; Q=$W/runs/glm53-flash/quick; L=$1; shift; D=$Q/$L; mkdir -p $D; E=${ENGINE_DIR:-$W/engine}
export CROW_CNQ='C:\Users\robin\dev\crow-nest\converter\GLM-5.3-Flash-MUL1K3.cnq'
TOK='C:\Users\robin\dev\crow-nest\models\GLM-5.3-Flash-original\tokenizer.json'; export CROW_TOKENIZER="$TOK"
# Operating set (robin 2026-10-10): ARM2 from arm2.env + CROW_GLM_RT2=1; no NVMe pool, controller or LA.
ARM="$(cat $Q/arm2.env) CROW_GLM_RT2=1"
cd $E
echo "start $(date +%T) head $(git rev-parse --short HEAD) builds $(tasklist | grep -ciE 'cargo|rustc') extra: $*" | tee $D/run.log
env $ARM "$@" timeout 150 target/release/glm5_run --cold --reps 1 --prompt-ids $Q/prefill8192.ids -n 4 --json $D/prefill.json > $D/prefill.log 2>&1; echo "prefill rc=$?" | tee -a $D/run.log
env $ARM "$@" timeout 120 target/release/glm5_run --cold --reps 1 --tokenizer "$TOK" --prompt "Explain in detail how a modern CPU cache hierarchy works: L1, L2, L3, coherence protocols, write-back versus write-through, and how false sharing hurts multithreaded code. Give concrete examples." -n 128 --json $D/decode.json > $D/decode.log 2>&1; echo "decode rc=$?" | tee -a $D/run.log
grep -hE "rep 1/1 .*(prefill:|decode:)" $D/prefill.log $D/decode.log | sed -E 's/.*(prefill|decode): /\1: /' | tee -a $D/run.log
echo "done $(date +%T)" | tee -a $D/run.log
