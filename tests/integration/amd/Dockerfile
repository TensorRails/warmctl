# A stand-in for the ROCm runtime image on a CPU host (make amd-e2e-test): the
# pinned DMTCP and libkfdreplay releases as the runtime image installs them,
# warmctl, the stand-in vLLM in its AMD mode, and a KFD topology of MI350X
# GPUs for warmctl to find (WARMCTL_TEST_KFD_ROOT). Only the GPUs are fake.
FROM debian:bookworm-slim@sha256:88200866dfff7ea7f5cbcb6ec7c8a701889efe6fe859fe64d6990e4b07ea4171
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 curl ca-certificates libstdc++6 libatomic1 procps && rm -rf /var/lib/apt/lists/*
COPY scripts/fetch-dmtcp.sh scripts/fetch-kfdreplay.sh /src/scripts/
COPY internal/dmtcp/VERSION internal/dmtcp/SHA256SUMS /src/internal/dmtcp/
COPY internal/kfdreplay/VERSION internal/kfdreplay/SHA256SUMS /src/internal/kfdreplay/
RUN /src/scripts/fetch-dmtcp.sh amd64 /opt/warmctl/dmtcp && /src/scripts/fetch-kfdreplay.sh /opt/warmctl/kfdreplay
RUN mkdir -p /opt/fake-rocm && head -c 65536 /dev/urandom > /opt/fake-rocm/libamdhip64.so.7 \
    && printf 'warmctl=amd-e2e\nbase=warmctl-amd-e2e:local\n' > /opt/warmctl/RUNTIME
COPY tests/integration/e2e/vllm.py /usr/local/bin/vllm
COPY bin/amd-e2e/kfd /opt/fake-kfd
COPY bin/amd-e2e/warmctl /usr/local/bin/warmctl
ENV WARMCTL_E2E_AMD=1 WARMCTL_TEST_KFD_ROOT=/opt/fake-kfd/a
