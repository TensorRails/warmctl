# vLLM 0.33 for ROCm 10.0

This Dockerfile builds an Ubuntu 24.04 image with a
Python 3.12 venv (`/opt/venv`, on `PATH`) holding `vllm==0.33.0+rocm100`,
`torch==2.13.0+rocm10.0.0` and the ROCm 10.0 (TheRock) wheels they bring, every
wheel pinned with its hash in `requirements.lock`. It is the base of warmctl's
AMD runtime image for vLLM 0.33 (`deploy/rocm/Dockerfile`), which AMD snapshots
of the Qwen3.8-27B and Qwen3.8-Flash-Next models are taken and restored in.

## Local build

Run both commands with the same Linux/amd64 Docker builder, from the
repository root. The runtime marker records the exact local base image ID.

```bash
base=warmctl-vllm-rocm:0.33.0-rocm100
docker build -t "$base" deploy/images/vllm-rocm-0.33
base_id=$(docker image inspect --format '{{.Id}}' "$base")
docker build -f deploy/rocm/Dockerfile --build-arg BASE="$base" \
  --build-arg BASE_REF="$base_id" -t warmctl-runtime-rocm:qwen38 .
```

Use that runtime for both `warmctl build` and `warmctl up`. To verify a
manually assembled venv, run this with the Python that launches vLLM.

```bash
python -c 'from importlib.metadata import version; assert version("vllm") == "0.33.0+rocm100"; assert version("torch") == "2.13.0+rocm10.0.0"'
```

## Updating the lock and publishing

`requirements.in` names the three pins that matter; `torchvision` is pinned
because the AMD index also serves 0.29, built for torch 2.14, whose operators
fail to register on torch 2.13 (`operator torchvision::nms does not exist`) and
take `transformers` down with them. Recompile the lock after changing it:

```bash
docker run --rm -v "$PWD":/w -w /w ghcr.io/astral-sh/uv:0.9.30-python3.12-bookworm \
  uv pip compile requirements.in -o requirements.lock --python-version 3.12 \
  --python-platform x86_64-manylinux_2_39 --index-url https://pypi.org/simple \
  --extra-index-url https://wheels.vllm.ai/rocm/0.33.0/rocm100/ --extra-index-url https://download.pytorch.org/whl/rocm10.0 \
  --extra-index-url https://stable.repo.amd.com/rocm/whl-next/ --index-strategy unsafe-best-match --generate-hashes
```

Build and publish as `RELEASING.md` publishes the runtime image (the registry
takes large layers only in chunks), then the runtime image on it, recording
the base by digest:

```bash
docker build -t localhost:5000/warmctl/vllm-rocm:0.33.0-rocm100 deploy/images/vllm-rocm-0.33
docker push localhost:5000/warmctl/vllm-rocm:0.33.0-rocm100
D=$(regctl image digest localhost:5000/warmctl/vllm-rocm:0.33.0-rocm100)
regctl image copy localhost:5000/warmctl/vllm-rocm:0.33.0-rocm100 registry.warmctl.io/warmctl/vllm-rocm:0.33.0-rocm100

docker build -f deploy/rocm/Dockerfile --build-arg VERSION=vX.Y.Z \
  --build-arg BASE=localhost:5000/warmctl/vllm-rocm:0.33.0-rocm100@$D \
  --build-arg BASE_REF=registry.warmctl.io/warmctl/vllm-rocm:0.33.0-rocm100@$D \
  -t localhost:5000/warmctl/runtime-rocm:vX.Y.Z-vllm0.33.0-rocm100 .
```

The image has no ROCm installation of its own: the wheels carry it
(`_rocm_sdk_core`). `_defaults-rocm.yaml` turns off what that ROCm loads by
default and a checkpoint cannot stop (rocprofiler-sdk) and points its JIT
caches under `/workspace`.

## Verify the complete images in GitHub Actions

The Linux/amd64 runner builds the wheel-pinned base and ROCm runtime in its
local Docker engine and runs CPU smoke checks. Both images expire with the
runner. The marker records the base image ID.

```bash
gh workflow run runtime-image.yml --repo TensorRails/warmctl --ref amd-rocm-backend \
  -f rocm_verify_only=true
```

## Build a Linux/amd64 staging image in GitHub Actions

Dispatch the runtime workflow from the ref containing the intended CLI and
libkfdreplay pin. Prepare the base ahead of the release, then pass the
`base=` reference from its job summary to the runtime build.

```bash
gh workflow run runtime-image.yml --repo TensorRails/warmctl --ref amd-rocm-backend \
  -f rocm_candidate_tag=staging-pr78-qwen38-base-001 -f rocm_base_only=true

gh workflow run runtime-image.yml --repo TensorRails/warmctl --ref amd-rocm-backend \
  -f rocm_candidate_tag=staging-pr78-qwen38-runtime-001 \
  -f base="registry.warmctl.io/warmctl/vllm-rocm@sha256:$DIGEST"
```

Use unique `staging-*` tags. The second build checks the pinned base digest,
recipe, wheel lock and Python versions. Its first changelog section supplies
the CLI version. Before publication it checks the CLI, runtime marker,
DMTCP and libkfdreplay on the CPU runner. The job summary and the
`rocm-candidate` artifact record the source commit and verified pull references.
Use the runtime reference by digest for the GPU bake and restore.
