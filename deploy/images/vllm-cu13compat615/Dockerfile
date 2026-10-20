# The vLLM DeepSeek-V4.1-Flash B300 recipe image with the CUDA 13.4 forward-compatible
# user-mode driver (libcuda 615.71.09) in place of 580.95.05: an r580 libcuda cannot
# restore multicast objects after cuda-checkpoint (ai-dynamo/snapshot#510).
ARG BASE=vllm/vllm-openai:nightly-x86_64-dev-20261012-59e75f4@sha256:db8bfec3185d4fabfec0f9434b72a408e9c8089ceb515eb6d3caaa3580cbe718
ARG COMPAT=vllm/vllm-openai:cu134-nightly@sha256:ad9dc6397abc878d8e764daa39e16bdc7f1ecf9fb3d0283c076530b299c31ad0
FROM ${COMPAT} AS compat
FROM ${BASE}
RUN rm -rf /usr/local/cuda-13.0/compat
COPY --from=compat /usr/local/cuda-13.4/compat/ /usr/local/cuda-13.0/compat/
LABEL io.warmctl.base="vllm/vllm-openai:nightly-x86_64-dev-20261012-59e75f4" io.warmctl.libcuda="615.71.09"
