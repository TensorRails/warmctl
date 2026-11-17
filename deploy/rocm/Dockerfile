# The warmctl runtime image for AMD Instinct GPUs: the ROCm vLLM image a
# snapshot's engine ran in, with warmctl, its vLLM extension, DMTCP,
# libkfdreplay (docs/amd.md) and tini on top. Nothing of the base changes, so
# every file the restored processes map is where the donor had it; the
# marker /opt/warmctl/RUNTIME makes `build` and `up` run in place:
#
#   tini -- warmctl build --profile qwen3-8b-mi350x-tp1 --model-dir /models/qwen3-8b --output /snap/q8
#   tini -- warmctl up /snap/q8
#
# Build from the repository root:
#
#   docker build -f deploy/rocm/Dockerfile -t warmctl-runtime-rocm .
#
# For vLLM 0.33 (ROCm 10.0 wheels), on deploy/images/vllm-rocm-0.33:
#
#   docker build -f deploy/rocm/Dockerfile \
#     --build-arg BASE=registry.warmctl.io/warmctl/vllm-rocm:0.33.0-rocm100@sha256:<digest> .
#
# BASE is the ROCm vLLM image AMD snapshots were verified with
# (profiles/_defaults-rocm.yaml); a restore refuses a snapshot taken in
# another one.
ARG BASE=rocm/vllm:rocm10.1.0_ubuntu24.04_py3.14-pytorch_2.13.0_vllm-0.29.0@sha256:e0ea9a10b729573ea8598bdc073bda0ea382fe1037eec7c0761b6fa7f46baaf4
ARG GO_IMAGE=golang:1.26-bookworm@sha256:23ce71337b726608afb1bc00986e630a05034a87a87ea1fc2cd1ae5cf33387eb

FROM ${GO_IMAGE} AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
ARG VERSION=dev
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -trimpath -ldflags "-X main.version=${VERSION}" -o /out/warmctl ./cmd/warmctl

# The released DMTCP+fdkeep and libkfdreplay, checked against the checksums
# pinned in this tree, and executed so that an incompatible BASE fails here.
FROM ${BASE} AS native
COPY scripts/fetch-dmtcp.sh scripts/fetch-kfdreplay.sh /src/scripts/
COPY internal/dmtcp/VERSION internal/dmtcp/SHA256SUMS /src/internal/dmtcp/
COPY internal/kfdreplay/VERSION internal/kfdreplay/SHA256SUMS /src/internal/kfdreplay/
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl libstdc++6 libatomic1 \
    && sh /src/scripts/fetch-dmtcp.sh amd64 /opt/warmctl/dmtcp \
    && sh /src/scripts/fetch-kfdreplay.sh /opt/warmctl/kfdreplay \
    && /opt/warmctl/dmtcp/bin/dmtcp_launch --version \
    && LD_PRELOAD=/opt/warmctl/kfdreplay/$(cat /src/internal/kfdreplay/VERSION)/libkfdreplay.so /bin/true

FROM ${BASE}
RUN apt-get update && apt-get install -y --no-install-recommends --no-upgrade libstdc++6 libatomic1 \
    && rm -rf /var/lib/apt/lists/*
ARG BASE
ARG BASE_REF=${BASE}
ARG VERSION=dev
# PID 1: reaps the restored engine.
ADD --chmod=0755 --checksum=sha256:c5b0666b4cb676901f90dfcb37106783c5fe2077b04590973b885950611b30ee \
    https://github.com/krallin/tini/releases/download/v0.19.0/tini-static-amd64 /usr/local/bin/tini
COPY --from=build /out/warmctl /usr/local/bin/warmctl
# COPY keeps the files' modification times: a DMTCP restart maps
# libkfdreplay and DMTCP's libraries again only from the very same files.
COPY --from=native /opt/warmctl/dmtcp /opt/warmctl/dmtcp
COPY --from=native /opt/warmctl/kfdreplay /opt/warmctl/kfdreplay
ENV PATH=/opt/warmctl/dmtcp/bin:${PATH}
COPY python/warmctl_vllm /opt/warmctl/adapter/warmctl_vllm
# Its vLLM plugin entry point (warmctl_vllm.plugin), as warmctl places it.
COPY python/warmctl_vllm-2.0.0a0.dist-info /opt/warmctl/adapter/warmctl_vllm-2.0.0a0.dist-info
# /tmp starts empty: an in-place run claims it only empty or as an earlier
# run left it. Last, the marker: this is the runtime image, built FROM BASE_REF.
RUN mkdir -p /workspace /run/warmctl /var/lib/warmctl \
    && rm -rf /tmp/* /tmp/.[!.]* \
    && printf 'warmctl=%s\nbase=%s\n' "${VERSION}" "${BASE_REF}" > /opt/warmctl/RUNTIME
ENTRYPOINT ["/usr/local/bin/tini", "--", "/usr/local/bin/warmctl"]
CMD ["up"]
