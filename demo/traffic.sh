#!/usr/bin/env bash
# Steady traffic through the Service while a pod is killed: one completion
# every half second from a small client pod, one line per request, and a
# running count of failures. Ctrl-C to stop; the client pod stays for reuse.
#   demo/traffic.sh
set -u
NS=${NS:-warmctl-demo}
SVC=${SVC:-glm53-flash}
kubectl -n "$NS" get pod warmctl-client >/dev/null 2>&1 || {
  kubectl -n "$NS" run warmctl-client --image=curlimages/curl:8.10.1 --restart=Never --command -- sleep infinity >/dev/null
  kubectl -n "$NS" wait --for=condition=Ready pod/warmctl-client --timeout=120s >/dev/null
}
kubectl -n "$NS" exec -i warmctl-client -- sh -s "$SVC" <<'EOF'
svc=$1; ok=0; fail=0
while true; do
  out=$(curl -s -m 30 -o /tmp/r -w '%{http_code} %{time_total}' "http://$svc/v1/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$svc\",\"prompt\":\"Count from one to five:\",\"max_tokens\":16,\"temperature\":0}")
  code=${out%% *}; secs=${out##* }
  if [ "$code" = 200 ]; then ok=$((ok+1)); mark=ok; else fail=$((fail+1)); mark=FAIL; fi
  echo "$(date +%H:%M:%S)  $mark  http $code  ${secs}s   total ok=$ok failed=$fail"
  sleep 0.5
done
EOF
