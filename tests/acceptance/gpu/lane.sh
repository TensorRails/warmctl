#!/usr/bin/env bash
# One GPU test lane: snapshot, then RESTORES warm restores, each answering a
# completion and a quality probe, then stopped. Lanes on disjoint GPUs run in
# parallel; a GPU lease makes a lane on busy GPUs fail at preflight.
#
#   lane.sh ID PROFILE MODEL_DIR GPUS PORT RESTORES [snapshot flags...]
#
# Env: RUN_DIR (logs and the binary, default /var/lib/warmctl-run),
# WARMCTL (binary, default $RUN_DIR/warmctl), STATE (state dir), OUTDIR
# (snapshot parent; keep it on local disk, never on a network share), PROBE,
# PROBE_TIMEOUT (seconds, default 600).
# Writes $RUN_DIR/diag/ID/ and one line per step to $RUN_DIR/lanes.log;
# timings.py turns the logs into per-stage timings.
set -u
ID=$1 PROFILE=$2 MODEL=$3 GPUS=$4 PORT=$5 RESTORES=$6
shift 6
R=${RUN_DIR:-/var/lib/warmctl-run}
B=${WARMCTL:-$R/warmctl}
STATE=${STATE:-/var/lib/warmctl}
OUT=${OUTDIR:-$R/snap}/$ID
PROBE=${PROBE:-/var/lib/warmctl/bin/quality-probe.py}
D=$R/diag/$ID
mkdir -p "$D" "$(dirname "$OUT")"
log() { echo "$(date -u +%T) $ID $*" | tee -a "$D/summary.txt" >> "$R/lanes.log"; }

log "snapshot start profile=$PROFILE gpus=$GPUS"
SECONDS=0
timeout 2400 "$B" snapshot --profile "$PROFILE" --id "$ID" --model-dir "$MODEL" --gpus "$GPUS" \
  --output "$OUT" --state-dir "$STATE" --progress plain "$@" > "$D/snapshot.log" 2>&1
rc=$?
log "snapshot exit $rc in ${SECONDS}s size=$(du -sh --apparent-size "$OUT" 2>/dev/null | cut -f1)"
[ $rc = 0 ] || exit $rc

for r in $(seq 1 "$RESTORES"); do
  RID=$ID-r$r L=$D/$ID-r$r.log
  SECONDS=0
  "$B" restore "$OUT" --id "$RID" --gpus "$GPUS" --listen 127.0.0.1:"$PORT" --state-dir "$STATE" \
    --progress plain > "$L" 2>&1 &
  P=$!
  while kill -0 $P 2>/dev/null && [ $SECONDS -lt 900 ]; do
    grep -q 'serving started' "$L" && break
    sleep 0.2
  done
  ok=0
  if grep -q 'serving started' "$L"; then
    name=$(curl -s --max-time 30 "http://127.0.0.1:$PORT/v1/models" |
      python3 -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["id"])')
    code=$(curl -s --max-time 120 -o "$D/$RID-completion.json" -w '%{http_code}' \
      "http://127.0.0.1:$PORT/v1/completions" -H 'Content-Type: application/json' \
      -d "{\"model\":\"$name\",\"prompt\":\"The capital of France is\",\"max_tokens\":16,\"temperature\":0}")
    timeout "${PROBE_TIMEOUT:-600}" python3 -I "$PROBE" "http://127.0.0.1:$PORT" "$name" "$D/$RID-probe.json" > "$D/$RID-probe.log" 2>&1
    probe=$?
    [ "$code" = 200 ] && [ $probe = 0 ] && ok=1
    log "$RID $(grep -o 'serving started.*' "$L" | head -1) | $(grep -o 'first host-gateway token.*' "$L" | head -1) | completion=$code | probe=$probe"
  else
    log "$RID serving not reached: $(tail -n 3 "$L" | tr '\n' ' ')"
  fi
  "$B" stop "$RID" --state-dir "$STATE" > "$D/$RID-stop.txt" 2>&1
  for _ in $(seq 1 120); do kill -0 $P 2>/dev/null || break; sleep 1; done
  wait $P
  log "$RID ok=$ok"
  [ $ok = 1 ] || exit 1
done
log "lane done"
