#!/bin/bash
# A/B of the CUDA restore's execution modes on a snapshot pulled from a
# registry:
#   cuda-ab-up.sh REF ID GPUS PORT ROUNDS
# The first up pulls REF into the store; every later one restores from it.
# ROUNDS ups per mode, alternating the modes; logs the "CUDA restored" line
# with its phase seconds, the fingerprints, the canary and the serving time.
# Env: RUN_DIR, WARMCTL, STATE as in lane.sh; MODES (default
# "auto parallel-all"); UP_FLAGS (extra flags for up).
set -u
REF=$1 ID=$2 GPUS=$3 PORT=$4 ROUNDS=$5
R=${RUN_DIR:-/var/lib/warmctl-run}
B=${WARMCTL:-$R/warmctl}
STATE=${STATE:-/var/lib/warmctl}
MODES=${MODES:-auto parallel-all}
D=$R/diag/$ID
mkdir -p "$D"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$D/ab.log"; }

for r in $(seq 1 "$ROUNDS"); do
  for mode in $MODES; do
    RID=$ID-$mode-r$r L=$D/$RID.log
    SECONDS=0
    # shellcheck disable=SC2086
    "$B" up "$REF" --id "$RID" --gpus "$GPUS" --listen 127.0.0.1:"$PORT" --state-dir "$STATE" \
      --cuda-execution "$mode" --progress plain ${UP_FLAGS:-} > "$L" 2>&1 &
    P=$!
    while kill -0 $P 2>/dev/null && [ $SECONDS -lt 3600 ]; do
      grep -q 'serving started\|up completed\|up failed' "$L" && break
      sleep 0.5
    done
    code=
    if grep -q 'serving started' "$L"; then
      name=$(curl -s --max-time 30 "http://127.0.0.1:$PORT/v1/models" |
        python3 -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["id"])')
      code=$(curl -s --max-time 120 -o "$D/$RID-completion.json" -w '%{http_code}' \
        "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' \
        -d "{\"model\":\"$name\",\"prompt\":\"The capital of France is\",\"max_tokens\":16,\"temperature\":0}")
    fi
    log "$RID | $(grep -o 'CUDA restored.*' "$L" | head -1) | $(grep -o 'CRIU restore in [0-9.]*s' "$L" | head -1) | $(grep -o 'fingerprints match\|fingerprint[^|]*mismatch[^|]*' "$L" | head -1) | $(grep -o 'canary[^|]*' "$L" | head -1) | $(grep -o 'serving started.*' "$L" | head -1) | completion=${code:-none} | $(grep -o 'up failed.*' "$L" | head -1)"
    "$B" down "$RID" --state-dir "$STATE" > "$D/$RID-down.txt" 2>&1
    for _ in $(seq 1 180); do kill -0 $P 2>/dev/null || break; sleep 1; done
    wait $P
    sleep 3
  done
done
log "done"
