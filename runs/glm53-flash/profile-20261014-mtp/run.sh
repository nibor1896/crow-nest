#!/bin/bash
# profile-20261014-mtp (#192, robin's GPU go): where the ~55 ms of a verify step go.
# One arm: nsys-mtp = quick/arm4.env + CROW_GLM_MTP=1 under nsys (verify step budget). The MTP-off reference is the
# existing profile-20261010l-decode (budget3.txt, same prompt and length); no new MTP-off run (robin, 2026-10-14).
# The other arm names stay available by hand only.
# Prompt and length = profile-20261010l-decode/run.sh (57-id question, -n 128, --cold --reps 1).
# Waits for <home>\dev\.gpu-lock, for no glm5_run/serve/decode/llama-server/sd-server process and for > 50.5 GiB
# free RAM (poll 60 s); holds the lock for the whole job; rebuilds glm5_run only if older than the engine sources.
# nsys: CUDA + NVTX trace, no CPU sampling, no context switches; glm5_run exits by itself (no Stop-Process, CUPTI tail kept).
# Usage: run.sh [arm ...]   (default: nsys-mtp)   DRY=1 run.sh  checks inputs, prints commands
W=<home>/dev; R=$W/crow-nest; D=$R/runs/glm53-flash/profile-20261014-mtp; E=$R/engine
LOCK=$W/.gpu-lock
NSYS="/c/Program Files/NVIDIA Corporation/Nsight Systems 2026.5.1/target-windows-x64/nsys.exe"
export CROW_CNQ='<home>\dev\crow-nest\converter\GLM-5.3-Flash-MUL1K3.cnq'
export CROW_GLM_MTP_OVERLAY='<home>\dev\crow-nest\converter\GLM-5.3-Flash-MTP-overlay.cnq'
TOK='<home>\dev\crow-nest\models\GLM-5.3-Flash-original\tokenizer.json'; export CROW_TOKENIZER="$TOK"
ARM4="$(cat $R/runs/glm53-flash/quick/arm4.env)"
P="Explain in detail how a modern CPU cache hierarchy works: L1, L2, L3, coherence protocols, write-back versus write-through, and how false sharing hurts multithreaded code. Give concrete examples."
ARMS=("$@"); [ ${#ARMS[@]} -eq 0 ] && ARMS=(nsys-mtp)
GLM=(target/release/glm5_run --cold --reps 1 --tokenizer "$TOK" --prompt "$P" -n 128)
log() { echo "$*" | tee -a $D/run.log; }
free_gib() { powershell -NoProfile -Command "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,2)" | tr -d '\r' | tr ',' '.'; }
busy() { tasklist 2>/dev/null | grep -qiE "^(glm5_run|serve|decode|llama-server|sd-server)\.exe"; }
mtp_env() { case "$1" in *-mtp) echo CROW_GLM_MTP=1;; *) echo;; esac; }

# ---- inputs (also the DRY self-test)
miss=0
for f in "$E/target/release/glm5_run.exe" "$NSYS" "$R/runs/glm53-flash/quick/arm4.env" \
         "$R/converter/GLM-5.3-Flash-MUL1K3.cnq" "$R/converter/GLM-5.3-Flash-MTP-overlay.cnq" \
         "$R/models/GLM-5.3-Flash-original/tokenizer.json" \
         "$(echo "$ARM4" | tr ' ' '\n' | sed -n 's/^CROW_GLM_ARENA_WARM=//p')"; do
  [ -e "$f" ] || { echo "missing: $f"; miss=1; }
done
for a in "${ARMS[@]}"; do case "$a" in plain-off|plain-mtp|nsys-off|nsys-mtp) ;; *) echo "unknown arm: $a"; miss=1;; esac; done
[ $miss -eq 0 ] || exit 1
if [ -n "$DRY" ]; then
  cd $E || exit 1
  echo "DRY: inputs ok; binary $(date -r target/release/glm5_run.exe '+%F %T') head $(git rev-parse --short HEAD) dirty $(git status --porcelain -- . | wc -l)"
  echo "DRY: newer sources than the binary: $(find src Cargo.toml build.rs -newer target/release/glm5_run.exe -type f 2>/dev/null | wc -l)"
  echo "DRY: free RAM $(free_gib) GiB (gate 50.5), VRAM used $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) MiB, lock $([ -d $LOCK ] && echo held || echo free), engines $(busy && echo running || echo none)"
  for a in "${ARMS[@]}"; do
    case "$a" in
      plain-*) echo "DRY $a: env -u CROW_GLM_MTP <arm4> $(mtp_env $a) timeout 600 ${GLM[*]} --json $D/$a.json > $D/$a.log";;
      nsys-*)  echo "DRY $a: env -u CROW_GLM_MTP <arm4> $(mtp_env $a) nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none -o $D/$a ${GLM[*]} --json $D/$a.json; nsys export --type sqlite -o $D/$a.sqlite";;
    esac
  done
  exit 0
fi

# ---- lock, engines, RAM
log "wait start $(date +%T) arms: ${ARMS[*]}"
while :; do
  if [ ! -d $LOCK ] && ! busy && awk -v f="$(free_gib)" 'BEGIN { exit !(f > 50.5) }'; then
    if mkdir $LOCK 2>/dev/null; then break; fi
  fi
  sleep 60
done
trap 'rmdir $LOCK && log "lock removed $(date +%T)"' EXIT
log "lock taken $(date +%T) free RAM $(free_gib) GiB"
cd $E || exit 1
if [ -n "$(find src Cargo.toml build.rs -newer target/release/glm5_run.exe -type f 2>/dev/null | head -1)" ]; then
  log "build glm5_run $(date +%T)"
  cargo build --release --bin glm5_run > $D/build.log 2>&1 || { log "build failed"; exit 1; }
fi
log "binary $(date -r target/release/glm5_run.exe '+%F %T') head $(git rev-parse --short HEAD) dirty $(git status --porcelain -- . | wc -l)"

for a in "${ARMS[@]}"; do
  while ! awk -v f="$(free_gib)" 'BEGIN { exit !(f > 50.5) }'; do sleep 30; done
  log "== $a start $(date +%T) free RAM $(free_gib) GiB vram_used_mib $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) env-extra: $(mtp_env $a)"
  case "$a" in
    plain-*)
      env -u CROW_GLM_MTP $ARM4 $(mtp_env $a) timeout 600 "${GLM[@]}" --json $D/$a.json > $D/$a.log 2>&1
      log "== $a rc=$? done $(date +%T)";;
    nsys-*)
      env -u CROW_GLM_MTP $ARM4 $(mtp_env $a) timeout 900 "$NSYS" profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
          --force-overwrite=true -o $D/$a "${GLM[@]}" --json $D/$a.json > $D/$a.log 2>&1
      log "== $a nsys rc=$? done $(date +%T)"
      "$NSYS" export --type sqlite --force-overwrite=true -o $D/$a.sqlite $D/$a.nsys-rep > /dev/null 2>&1
      log "== $a export rc=$?";;
  esac
  grep -hE "rep 1/1 .*(prefill:|decode:)|decode MTP:|refused" $D/$a.log | tee -a $D/run.log
done
log "all done $(date +%T)"
