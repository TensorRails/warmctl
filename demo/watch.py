#!/usr/bin/env python3
"""Live view of the demo's replicas: node, state, seconds since the pod was
created, what the restore is doing, and, once a pod is ready, its first answer
and time to first token. Needs only python3 and kubectl.

    demo/watch.py                     # namespace warmctl-demo, app glm53-flash
    NS=other APP=other demo/watch.py
"""
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

NS = os.environ.get("NS", "warmctl-demo")
APP = os.environ.get("APP", "glm53-flash")
PROMPT = os.environ.get("PROMPT", "The capital of France is")

# Runs inside the pod: one streamed completion against the pod's own gateway.
PROBE = r"""
import json, time, urllib.request
body = json.dumps({"model": "%s", "prompt": %s, "max_tokens": 8, "temperature": 0, "stream": True}).encode()
req = urllib.request.Request("http://127.0.0.1:8080/v1/completions", body, {"Content-Type": "application/json"})
t0 = time.time(); ttft = None; text = ""
with urllib.request.urlopen(req, timeout=60) as r:
    for raw in r:
        line = raw.decode().strip()
        if not line.startswith("data:") or line.endswith("[DONE]"):
            continue
        piece = json.loads(line[5:])["choices"][0].get("text", "")
        if piece and ttft is None:
            ttft = time.time() - t0
        text += piece
print(json.dumps({"ttft": ttft, "text": text.strip(), "total": time.time() - t0}))
"""

answers, logs, lock = {}, {}, threading.Lock()


def sh(*args, timeout=20):
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout)


def ts(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()


def probe(pod):
    script = PROBE % (APP, json.dumps(PROMPT))
    try:
        r = sh("kubectl", "-n", NS, "exec", pod, "-c", "warmctl", "--", "python3", "-c", script, timeout=90)
        res = json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 else {"error": r.stderr.strip()[-80:]}
    except Exception as e:  # noqa: BLE001
        res = {"error": str(e)[-80:]}
    with lock:
        answers[pod] = res


def last_log(pod):
    try:
        r = sh("kubectl", "-n", NS, "logs", pod, "-c", "warmctl", "--tail=1", timeout=10)
        line = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
    except Exception:  # noqa: BLE001
        line = ""
    with lock:
        logs[pod] = line


def main():
    done = []          # (pod, node, seconds to ready) for every pod that got ready
    tick = 0
    while True:
        r = sh("kubectl", "-n", NS, "get", "pods", "-l", f"app={APP}", "-o", "json")
        if r.returncode:
            print(r.stderr.strip()); time.sleep(2); continue
        now = time.time()
        rows = []
        for p in sorted(json.loads(r.stdout)["items"], key=lambda p: p["metadata"]["creationTimestamp"]):
            name = p["metadata"]["name"]
            node = p["spec"].get("nodeName", "—")
            created = ts(p["metadata"]["creationTimestamp"])
            ready_cond = next((c for c in p["status"].get("conditions", []) if c["type"] == "Ready"), None)
            ready = ready_cond and ready_cond["status"] == "True"
            if p["metadata"].get("deletionTimestamp"):
                state = "Terminating"
            elif ready:
                state = "Ready"
            else:
                waiting = [s["state"].get("waiting", {}).get("reason") for s in p["status"].get("containerStatuses", [])]
                state = next((w for w in waiting if w), p["status"].get("phase", "?"))
                if state == "Running":
                    state = "Restoring"
            to_ready = ts(ready_cond["lastTransitionTime"]) - created if ready else None
            if ready and name not in answers:
                with lock:
                    answers[name] = None
                threading.Thread(target=probe, args=(name,), daemon=True).start()
                if all(d[0] != name for d in done):
                    done.append((name, node, to_ready))
            if not ready and state == "Restoring" and tick % 2 == 0:
                threading.Thread(target=last_log, args=(name,), daemon=True).start()
            rows.append((name, node, state, now - created, to_ready))
        tick += 1

        out = [f"\033[2J\033[H\033[1mwarmctl demo · {APP} · namespace {NS}\033[0m   {time.strftime('%H:%M:%S')}",
               f"ready {sum(1 for r in rows if r[2] == 'Ready')}/{len(rows)} replicas\n",
               f"{'POD':<34} {'NODE':<26} {'STATE':<12} {'AGE':>7} {'READY IN':>9}  DETAIL"]
        for name, node, state, age, to_ready in rows:
            color = {"Ready": "\033[32m", "Terminating": "\033[31m"}.get(state, "\033[33m")
            with lock:
                ans, log = answers.get(name), logs.get(name, "")
            if state == "Ready":
                if ans is None:
                    detail = "asking the model…"
                elif "error" in ans:
                    detail = "error: " + ans["error"]
                else:
                    detail = f"“{PROMPT}” → {ans['text']!r}  first token {ans['ttft'] * 1000:.0f} ms"
            else:
                detail = log[-70:]
            ready_s = f"{to_ready:.1f}s" if to_ready is not None else "—"
            out.append(f"{name:<34} {node[:26]:<26} {color}{state:<12}\033[0m {age:6.0f}s {ready_s:>9}  {detail}")
        if done:
            out.append("\n\033[1mtime from pod created to serving\033[0m")
            out += [f"  {n:<34} {nd[:26]:<26} {s:6.1f} s" for n, nd, s in done[-8:]]
        print("\n".join(out), flush=True)
        time.sleep(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
