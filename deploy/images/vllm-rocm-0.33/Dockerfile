# vLLM 0.33 for ROCm 10.0 (TheRock wheels) in a Python 3.12 venv on Ubuntu
# 24.04: the base of warmctl's AMD runtime image for vLLM 0.33
# (deploy/rocm/Dockerfile, BASE=registry.warmctl.io/warmctl/vllm-rocm:0.33.0-rocm100).
# requirements.lock pins every wheel with its hash (see README.md); the image
# has no ROCm of its own, the wheels carry it.
ARG UBUNTU=ubuntu:24.04@sha256:2198e4c65076e9cb0b3d695219df667c0e3b0072da64ca3c2648aca17dcecaf5
ARG UV=ghcr.io/astral-sh/uv:0.9.30@sha256:538e0b39736e7feae937a65983e49d2ab75e1559d35041f9878b7b7e51de91e4

FROM ${UV} AS uv

FROM ${UBUNTU}
# Triton and tvm-ffi compile host code at run time (gcc, Python headers);
# platform.architecture() runs `file`.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev gcc g++ libgomp1 libstdc++6 libatomic1 \
        file ca-certificates curl procps \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /usr/local/bin/uv
COPY requirements.lock /opt/vllm-rocm/requirements.lock
RUN uv venv -q -p /usr/bin/python3.12 /opt/venv \
    && VIRTUAL_ENV=/opt/venv uv pip sync -q --require-hashes \
        --index-url https://pypi.org/simple \
        --extra-index-url https://wheels.vllm.ai/rocm/ \
        --extra-index-url https://download.pytorch.org/whl/rocm10.0 \
        --extra-index-url https://stable.repo.amd.com/rocm/whl-next/ \
        --index-strategy unsafe-best-match \
        /opt/vllm-rocm/requirements.lock \
    && rm -rf /root/.cache/uv
ENV VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:${PATH}
# Fails the build on a broken pin (torchvision for another torch fails its
# operator registration on import).
RUN python -c "import torch, torchvision, transformers, vllm; \
assert torch.version.hip, torch.__version__; \
print('vllm', vllm.__version__, 'torch', torch.__version__, 'hip', torch.version.hip, 'torchvision', torchvision.__version__)"
LABEL io.warmctl.vllm="0.33.0+rocm100" io.warmctl.rocm="10.0.0" io.warmctl.torch="2.13.0+rocm10.0.0"
CMD ["vllm", "--help"]
