# The warmctl runtime image: the vLLM image a snapshot's engine ran in,
# with warmctl, its vLLM extension, NVIDIA's cuInterpose and tini on top.
# Nothing of the base changes, so every file the restored processes map is
# where the donor had it. The command is the one a host runs; the marker
# /opt/warmctl/RUNTIME makes `up` restore in place (docs/kubernetes.md):
#
#   tini -- warmctl up NAME|DIR
#
# Build from the repository root (make pod-image):
#
#   docker build -f deploy/kubernetes/Dockerfile --build-context cuinterpose=bin/cuinterpose .
#
# BASE must be the snapshot's image, manifest.json's launch.image; the
# restore refuses a snapshot taken in another one. BASE_REF is what the
# marker records as the base when BASE is a local name (the CPU test).
ARG BASE=vllm/vllm-openai:cu134-nightly@sha256:ad9dc6397abc878d8e764daa39e16bdc7f1ecf9fb3d0283c076530b299c31ad0
ARG GO_IMAGE=golang:1.26-bookworm@sha256:23ce71337b726608afb1bc00986e630a05034a87a87ea1fc2cd1ae5cf33387eb

FROM ${GO_IMAGE} AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
ARG VERSION=dev
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -trimpath -ldflags "-X main.version=${VERSION}" -o /out/warmctl ./cmd/warmctl

# cuInterpose's build (make cuinterpose), passed as a named build context;
# empty without one. A restore runs the snapshot's own copy, so only a bake
# in a pod would need it.
FROM scratch AS cuinterpose

FROM ${BASE}
ARG BASE
ARG BASE_REF=${BASE}
ARG VERSION=dev
# PID 1: reaps the restored engine, which CRIU leaves to it.
ADD --chmod=0755 --checksum=sha256:c5b0666b4cb676901f90dfcb37106783c5fe2077b04590973b885950611b30ee \
    https://github.com/krallin/tini/releases/download/v0.19.0/tini-static-amd64 /usr/local/bin/tini
COPY --from=build /out/warmctl /usr/local/bin/warmctl
# cuinterpose/ next to warmctl, where a release keeps it.
COPY --from=cuinterpose / /usr/local/bin/cuinterpose/
# The extension for engines started from this image; a restore mounts the
# snapshot's own copy over it.
COPY python/warmctl_vllm /opt/warmctl/adapter/warmctl_vllm
# Mount points of a read-only root: the restore bind-mounts the snapshot's
# cuInterpose build at /opt/warmctl/ipc, the pod mounts its volumes on the
# others. Last, the marker: this is the runtime image, built FROM BASE_REF.
RUN mkdir -p /opt/warmctl/ipc /workspace /run/warmctl /var/lib/warmctl \
    && printf 'warmctl=%s\nbase=%s\n' "${VERSION}" "${BASE_REF}" > /opt/warmctl/RUNTIME
ENTRYPOINT ["/usr/local/bin/tini", "--", "/usr/local/bin/warmctl"]
CMD ["up"]
