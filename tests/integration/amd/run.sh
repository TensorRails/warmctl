#!/bin/bash
# The AMD path of warmctl end to end on a CPU host (make amd-e2e-test): a bake
# and restores in unprivileged containers of a stand-in ROCm runtime image,
# as on a RunPod AMD pod. Real: warmctl, DMTCP and libkfdreplay (the pinned
# releases), the agent, the trees, the manifest, the gateway and warmctl's
# Python extension calling libkfdreplay. Fake: the GPUs (a KFD topology of
# MI350X GPUs, a stand-in vLLM whose "GPU memory" is host memory).
#
# Usage: tests/integration/amd/run.sh IMAGE
set -uo pipefail
image=${1:?usage: run.sh IMAGE}
pin=$(dirname "$0")/../../../internal/kfdreplay
lib_tag=$(cat "$pin/VERSION")
lib_version=${lib_tag#v}
lib_sha=$(awk '$2 == "libkfdreplay.so" {print $1}' "$pin/SHA256SUMS")
[[ $lib_sha =~ ^[0-9a-f]{64}$ ]] || exit 1
# Outside /tmp: the engine's /tmp is a tree warmctl claims, and the model
# must be at the same path in every container.
work=$(mktemp -d /var/tmp/warmctl-amd-e2e.XXXXXX)
chmod 755 "$work"
model=$work/models/fake
port=${AMD_E2E_PORT:-18091}
failed=0
names=()

cleanup() {
  if [ -n "${AMD_E2E_LOGS:-}" ]; then
    mkdir -p "$AMD_E2E_LOGS" && cp "$work"/*.txt "$work"/*.json "$AMD_E2E_LOGS"/ 2>/dev/null
    cp "$work/snap/s1/manifest.json" "$AMD_E2E_LOGS/manifest.json" 2>/dev/null
  fi
  for n in "${names[@]}"; do docker rm -f "$n" >/dev/null 2>&1; done
  docker run --rm -v "$work:/w" --entrypoint rm "$image" -rf /w/snap /w/state >/dev/null 2>&1
  rm -rf "$work"
}
trap cleanup EXIT

pass() { echo "PASS $*"; }
fail() { echo "FAIL $*"; failed=1; }
expect() { # name text-file pattern...
  local name=$1 file=$2; shift 2
  for p in "$@"; do grep -qE -- "$p" "$file" || { fail "$name: no /$p/ in output"; sed 's/^/    /' "$file" | tail -40; return 1; }; done
  pass "$name"
}

mkdir -p "$model" "$work/profiles" "$work/snap"
echo '{"seed": "amd-e2e"}' > "$model/config.json"
image_id=$(docker image inspect -f '{{.Id}}' "$image")
cat > "$work/profiles/fake-amd-tp1.yaml" <<YAML
schema: warmctl.profile/v1
id: fake-amd-tp1
image: $image_id
model: {id: fake/model, revision: "0", served_names: [fake]}
parallel: {tp: 1}
hardware: [mi350x-288gb]
warmup: {prefill_tokens: [64, 128], decode_prompt_tokens: 32}
canary: {mode: bitwise, prefix_tokens: 64}
resources: {memory: 2GiB, pids: 4096, shm: 64MiB, cpus: 2}
YAML
sed -e 's/^id: fake-amd-tp1/id: fake-amd-tp2/' -e 's/{tp: 1}/{tp: 2}/' \
  "$work/profiles/fake-amd-tp1.yaml" > "$work/profiles/fake-amd-tp2.yaml"

# box NAME KFD-FIXTURE ARGS...: one unprivileged container of the image, as a
# RunPod pod is (Docker's default capabilities and seccomp profile).
box() {
  local name=$1 kfd=$2; shift 2
  names+=("$name")
  docker run --name "$name" -e "WARMCTL_TEST_KFD_ROOT=/opt/fake-kfd/$kfd" \
    -v "$model:$model:ro" -v "$work/profiles:/profiles:ro" -v "$work/snap:/snap" "$@"
}

bake=(warmctl build --test-cpu --profile fake-amd-tp1 --profile-dir /profiles --model-dir "$model"
      --id amd-e2e --part-bytes 300000)

# 1. Bake on GPU a: the vendor comes from the host, in place without being asked.
box amd-e2e-bake a --rm "$image" "${bake[@]}" --output /snap/s1 > "$work/bake.txt" 2>&1
echo "bake exit $?" >> "$work/bake.txt"
expect "bake on an AMD host" "$work/bake.txt" \
  "AMD Instinct host" "in the runtime image: baking in place" "libkfdreplay $lib_version at /opt/warmctl/kfdreplay/$lib_tag" \
  "GPU state checkpointed by 1 workers \(libkfdreplay\)" "GPU devices released by every process: 1 run ROCm with libkfdreplay" "bake exit 0"

# 1b. doctor on the AMD host: what AMD needs, none of NVIDIA's checks.
box amd-e2e-doctor a --rm "$image" warmctl doctor > "$work/doctor.txt" 2>&1
expect "doctor on an AMD host" "$work/doctor.txt" "ok   gpus +1 x AMD Instinct MI350X \(gfx950\)" \
  "ok   amd-support" "ok   runtime-image" "ok   dmtcp" "ok   libkfdreplay +$lib_version"
if grep -qE "nvidia-runtime|io_uring|docker" "$work/doctor.txt"; then fail "doctor runs NVIDIA checks on AMD"; fi

# 2. The manifest: AMD schema and fields, no CUDA owner.
docker run --rm -i -v "$work/snap:/snap:ro" --entrypoint python3 "$image" - > "$work/manifest.txt" 2>&1 <<'PY'
import json
m = json.load(open('/snap/s1/manifest.json'))
h, p = m['hardware'], m['process']
print('schema', m['schema'])
print('vendor', h.get('vendor'), 'gpu', h['gpus'][0]['name'], h['gpus'][0].get('arch'), h['gpus'][0].get('kfd_gpu_id'),
      h['gpus'][0].get('device'))
print('backend', p.get('backend'), 'state', p.get('gpu_state'), 'cuda_owners', len(p.get('cuda_owners') or []),
      'gpu_owners', len(p.get('gpu_owners') or []))
print('lib', p['gpu_lib']['path'], p['gpu_lib']['version'], p['gpu_lib']['sha256'][:12], p['gpu_lib']['mtime_ns'] > 0)
print('checkpoint', h.get('gpu_checkpoint'))
print('driver', h['driver'], 'row', h['row']['id'])
PY
expect "AMD manifest" "$work/manifest.txt" \
  "schema warmctl.snapshot/v2-rocm" "vendor amd gpu AMD Instinct MI350X gfx950 41001 /dev/dri/renderD136" \
  "backend dmtcp state libkfdreplay cuda_owners 0 gpu_owners 1" \
  "lib /opt/warmctl/kfdreplay/$lib_tag/libkfdreplay.so $lib_version ${lib_sha:0:12} True" \
  "checkpoint libkfdreplay $lib_version sha256:$lib_sha" "driver 7.1.3.31500000 row mi350x-288gb"

# serve NAME KFD COMMAND...: start the command in place and wait for the gateway.
serve() {
  local name=$1 kfd=$2; shift 2
  box "$name" "$kfd" -d -p "127.0.0.1:$port:8080" "$image" "$@" >/dev/null
  for _ in $(seq 1 120); do
    curl -fsS "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1 && return 0
    [ "$(docker inspect -f '{{.State.Running}}' "$name")" = true ] || return 1
    sleep 1
  done
  return 1
}
complete() {
  curl -fsS "http://127.0.0.1:$port/v1/completions" -H 'Content-Type: application/json' \
    -d '{"model": "fake", "prompt": [1, 2, 3, 4, 5, 6, 7, 8], "max_tokens": 8, "logprobs": 1}'
}

# 3. Restore on the donor's GPU, then 4. on another GPU of the host (a new
# gpu_id and render node: libkfdreplay translates), each in a new container.
for case in "a:the donor's GPU" "b:another GPU"; do
  kfd=${case%%:*}; what=${case#*:}; name=amd-e2e-up-$kfd
  if serve "$name" "$kfd" warmctl up --test-cpu /snap/s1; then
    answer=$(complete); docker logs "$name" > "$work/up-$kfd.txt" 2>&1
    echo "answer $answer" >> "$work/up-$kfd.txt"
    expect "restore on $what" "$work/up-$kfd.txt" "GPU state restored by 1 workers \(libkfdreplay\)" \
      "canary" "answer .*token_id"
    echo "$answer" > "$work/answer-$kfd.json"
  else
    docker logs "$name" > "$work/up-$kfd.txt" 2>&1; fail "restore on $what"; tail -40 "$work/up-$kfd.txt" | sed 's/^/    /'
  fi
  docker rm -f "$name" >/dev/null 2>&1
done
if [ -s "$work/answer-a.json" ] && cmp -s "$work/answer-a.json" "$work/answer-b.json"; then
  pass "the same answer on both GPUs"
else
  fail "answers differ between the GPUs"
fi

# Repeat build and up in the same container, as on a pod. The stand-in
# engine has no extra ROCm owner; publish its late reply through the real
# plugin in the directory the baked engine was given.
cat > "$work/profiles/bake-up.sh" <<'SH'
set -eu
"$@" --output /snap/same
PYTHONPATH=/opt/warmctl/adapter python3 - <<'PY'
import json
import os
from warmctl_vllm import plugin
with open('/snap/same/manifest.json') as f:
    os.environ[plugin.STATUS_ENV] = json.load(f)['launch']['env'][plugin.STATUS_ENV]
plugin._report('checkpointed')
assert os.path.isfile(os.path.join(os.environ[plugin.STATUS_ENV], str(os.getpid())))
PY
exec warmctl up --test-cpu /snap/same
SH
name=amd-e2e-same-container
if serve "$name" a sh /profiles/bake-up.sh "${bake[@]}"; then
  answer=$(complete)
  docker logs "$name" > "$work/same-container.txt" 2>&1
  echo "answer $answer" >> "$work/same-container.txt"
  expect "same-container build then default up" "$work/same-container.txt" "serving started" "answer .*token_id"
else
  docker logs "$name" > "$work/same-container.txt" 2>&1
  fail "same-container build then default up"
  tail -40 "$work/same-container.txt"
fi
docker rm -f "$name" >/dev/null 2>&1

# refuse NAME KFD PATTERN [BOX-ARGS] -- [UP-ARGS]: a restore that must not start.
refuse() {
  local name=$1 kfd=$2 pattern=$3; shift 3
  box "amd-e2e-$name" "$kfd" --rm --entrypoint sh "$image" -c "timeout 180 $*" > "$work/$name.txt" 2>&1
  echo "exit $?" >> "$work/$name.txt"
  if grep -q "exit 0" "$work/$name.txt"; then fail "$name: accepted"; else expect "$name" "$work/$name.txt" "$pattern"; fi
}
refuse another-gpu-among-two ab "libkfdreplay moves a process to another GPU only when one is visible" \
  "warmctl up --test-cpu --gpus 1 /snap/s1"
refuse library-touched a "libkfdreplay.*mtime" \
  "touch /opt/warmctl/kfdreplay/$lib_tag/libkfdreplay.so; timeout 180 warmctl up --test-cpu /snap/s1"
refuse multi-visible-donor ab "AMD bake needs exactly one visible GPU render node, found 2" \
  "warmctl build --test-cpu --gpus 1 --profile fake-amd-tp1 --profile-dir /profiles --model-dir $model --id x --output /snap/x"
refuse mismatched-worker a "rank 0 reports PCI device.*0000:c1:00.0.*selected GPU is 0000:09:00.0" \
  "env WARMCTL_E2E_WORKER_PCI=0000:c1:00.0 warmctl build --test-cpu --profile fake-amd-tp1 --profile-dir /profiles --model-dir $model --id x --output /snap/x"
refuse two-gpus-per-engine a "one GPU per engine" \
  "warmctl build --test-cpu --profile fake-amd-tp2 --profile-dir /profiles --model-dir $model --id x --output /snap/x"

exit $failed
