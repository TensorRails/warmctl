# The CRIU test image plus Python and the stand-in vLLM (make e2e-test).
ARG BASE=warmctl-criu-test:local
FROM ${BASE}
RUN apt-get update && apt-get install -y --no-install-recommends python3 && rm -rf /var/lib/apt/lists/*
COPY vllm.py /usr/local/bin/vllm
