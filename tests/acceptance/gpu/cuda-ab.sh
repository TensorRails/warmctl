#!/bin/bash
# A/B of the CUDA restore's execution modes on one snapshot:
#   cuda-ab.sh ID PROFILE MODEL_DIR GPUS PORT ROUNDS [snapshot flags...]
# Builds the snapshot once (skipped when $OUTDIR/$ID exists), then restores it
# ROUNDS times per mode, alternating the modes, and logs the "CUDA restored"
# line with its phase seconds, the canary and the fingerprints of each.
# Env: RUN_DIR, WARMCTL, STATE, OUTDIR as in lane.sh; MODES (default
# "auto parallel-all"): each is a --cuda-execution value, optionally
# followed by up flags joined with "+", e.g. "auto+sync-weights" or
# "parallel-all+threads=8".
set -u
ID=$1 PROFILE=$2 MODEL=$3 GPUS=$4 PORT=$5 ROUNDS=$6
shift 6
R=${RUN_DIR:-/var/lib/warmctl-run}
B=${WARMCTL:-$R/warmctl}
STATE=${STATE:-/var/lib/warmctl}
OUT=${OUTDIR:-$R/snap}/$ID
MODES=${MODES:-auto parallel-all}
D=$R/diag/$ID
mkdir -p "$D" "$(dirname "$OUT")"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$D/ab.log"; }

if [ -f "$OUT/manifest.json" ]; then
  log "snapshot $OUT exists, reusing"
else
  # A build cut short (a preempted VM) leaves a partial directory behind.
  rm -rf "$OUT"
  SECONDS=0
  timeout 2400 "$B" snapshot --profile "$PROFILE" --id "$ID" --model-dir "$MODEL" --gpus "$GPUS" \
    --output "$OUT" --state-dir "$STATE" --progress plain "$@" > "$D/snapshot.log" 2>&1
  rc=$?
  log "snapshot exit $rc in ${SECONDS}s size=$(du -sh --apparent-size "$OUT" 2>/dev/null | cut -f1)"
  [ $rc = 0 ] || exit $rc
fi
python3 - "$OUT/manifest.json" <<'EOF' | tee -a "$D/ab.log"
import json, sys
p = json.load(open(sys.argv[1]))["process"]
o = p["cuda_owners"]
print("owners", len(o), "ranked", sum(1 for w in o if w["rank"] >= 0),
      "collectives_detached", p.get("collectives_detached"), "ipc", p.get("ipc"))
EOF

for r in $(seq 1 "$ROUNDS"); do
  for mode in $MODES; do
    IFS=+ read -r -a parts <<< "$mode"
    flags=()
    for f in "${parts[@]:1}"; do flags+=("--$f"); done
    RID=$ID-${mode//[+=]/-}-r$r L=$D/$RID.log
    # A rerun after a reboot skips the restores that already served.
    grep -q "^[^ ]* $RID .*serving started" "$D/ab.log" 2>/dev/null && continue
    SECONDS=0
    "$B" restore "$OUT" --id "$RID" --gpus "$GPUS" --listen 127.0.0.1:"$PORT" --state-dir "$STATE" \
      --cuda-execution "${parts[0]}" "${flags[@]}" --progress plain > "$L" 2>&1 &
    P=$!
    while kill -0 $P 2>/dev/null && [ $SECONDS -lt 900 ]; do
      grep -q 'serving started\|up completed\|restore completed\|error' "$L" && break
      sleep 0.2
    done
    code=
    if grep -q 'serving started' "$L"; then
      name=$(curl -s --max-time 30 "http://127.0.0.1:$PORT/v1/models" |
        python3 -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["id"])')
      code=$(curl -s --max-time 120 -o "$D/$RID-completion.json" -w '%{http_code}' \
        "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' \
        -d "{\"model\":\"$name\",\"prompt\":\"The capital of France is\",\"max_tokens\":16,\"temperature\":0}")
    fi
    log "$RID | $(grep -o 'CUDA restored.*' "$L" | head -1) | $(grep -o 'weights reloaded.*\|fingerprint[^|]*differs[^|]*' "$L" | head -1) | $(grep -o 'canary[^|]*' "$L" | head -1) | $(grep -o 'serving started.*' "$L" | head -1) | completion=${code:-none}"
    "$B" stop "$RID" --state-dir "$STATE" > "$D/$RID-stop.txt" 2>&1
    for _ in $(seq 1 120); do kill -0 $P 2>/dev/null || break; sleep 1; done
    wait $P
    sleep 3
  done
done
log "done"
