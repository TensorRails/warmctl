.PHONY: catalog-push demo build linux test pytest lint profiles-check check criu-test criu-image e2e-test cuinterpose pod-image release clean

GO ?= go
PY ?= python3
VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)
LDFLAGS := -X main.version=$(VERSION)
CRIU_IMAGE ?= warmctl-criu-test:local
E2E_IMAGE ?= warmctl-e2e:local
E2E_POD_IMAGE ?= warmctl-e2e-pod:local
POD_IMAGE ?= warmctl-pod:$(VERSION)
SUDO ?= $(shell [ "$$(id -u)" = 0 ] || echo sudo -E)
# cuInterpose (docs/interpose.md): NVIDIA's release v0.2.0-rc.2. Its build
# takes only cuda.h from CUDA_HEADERS_IMAGE; any CUDA 13 headers do.
CUINTERPOSE_REPO ?= https://github.com/ai-dynamo/snapshot.git
CUINTERPOSE_REF ?= 50bf0db3575df64c968e3f3255d5fa8938950b7a
CUDA_HEADERS_IMAGE ?= nvidia/cuda:13.3.0-devel-ubuntu24.04

build:
	$(GO) build -trimpath -ldflags "$(LDFLAGS)" -o bin/warmctl ./cmd/warmctl

linux:
	CGO_ENABLED=0 GOOS=linux GOARCH=amd64 $(GO) build -trimpath -ldflags "$(LDFLAGS)" -o bin/warmctl-linux-amd64 ./cmd/warmctl

test:
	$(GO) vet ./...
	$(GO) test -race ./...

# The adapter tests need no torch or vLLM; pytest when installed, unittest otherwise.
pytest:
	cd python && ($(PY) -m pytest -q tests 2>/dev/null || $(PY) -m unittest discover -s tests -t . -v)

lint:
	golangci-lint run ./...
	GOOS=linux golangci-lint run --build-tags criu,e2e ./tests/...
	ruff check python
	ruff check --config python/pyproject.toml tests/integration/e2e/vllm.py

# Every shipped profile must load, merge over the defaults, validate and encode.
profiles-check:
	$(GO) run ./cmd/warmctl profile check

# What CONTRIBUTING asks before a pull request.
check: test pytest profiles-check lint

criu-image:
	docker build -t $(CRIU_IMAGE) -f tests/integration/Dockerfile.criu tests/integration

# The process half of a snapshot on a CPU host: a donor container freezes,
# dumps and archives a stand-in engine; a receiver container restores it
# stopped, thaws it and checks its memory, files and socket. Needs Docker and
# root, since the snapshot files belong to the containers' root.
criu-test: criu-image
	mkdir -p bin/criu
	CGO_ENABLED=0 GOOS=linux GOARCH=$(shell $(GO) env GOARCH) $(GO) build -o bin/criu/warmctl ./cmd/warmctl
	CGO_ENABLED=0 GOOS=linux GOARCH=$(shell $(GO) env GOARCH) $(GO) build -o bin/criu/witness ./tests/integration/criu/witness
	$(GO) test -c -tags criu -o bin/criu/roundtrip.test ./tests/integration/criu
	$(SUDO) bin/criu/roundtrip.test -test.v -test.count=1 -bin $(CURDIR)/bin/criu -image $(CRIU_IMAGE)

# A whole snapshot and two restores through the real binary on a CPU host,
# around a stand-in vLLM (tests/integration/e2e/vllm.py) that runs warmctl's
# Python extension over emulated device memory, then restores in the pod
# image built on it, run as a Kubernetes pod would run it. Needs Docker and
# root.
e2e-test: criu-image
	docker build -t $(E2E_IMAGE) --build-arg BASE=$(CRIU_IMAGE) tests/integration/e2e
	docker build -t $(E2E_POD_IMAGE) -f deploy/kubernetes/Dockerfile --build-arg BASE=$(E2E_IMAGE) \
		--build-arg BASE_REF=$$(docker image inspect -f '{{.Id}}' $(E2E_IMAGE)) .
	mkdir -p bin/e2e
	CGO_ENABLED=0 GOOS=linux GOARCH=$(shell $(GO) env GOARCH) $(GO) build -o bin/e2e/warmctl ./cmd/warmctl
	$(GO) test -c -tags e2e -o bin/e2e/e2e.test ./tests/integration/e2e
	docker rm -f warmctl-e2e-registry >/dev/null 2>&1 || true
	docker run -d --name warmctl-e2e-registry -p 127.0.0.1:5000:5000 registry:2 >/dev/null
	$(SUDO) bin/e2e/e2e.test -test.v -test.count=1 -warmctl $(CURDIR)/bin/e2e/warmctl -image $(E2E_IMAGE) \
		-testdata $(CURDIR)/tests/integration/e2e/testdata -registry localhost:5000 -pod-image $(E2E_POD_IMAGE); \
	status=$$?; docker rm -f warmctl-e2e-registry >/dev/null; exit $$status

cuinterpose:
	rm -rf bin/cuinterpose bin/cuinterpose-src
	git clone --quiet --filter=blob:none $(CUINTERPOSE_REPO) bin/cuinterpose-src
	git -C bin/cuinterpose-src checkout --quiet $(CUINTERPOSE_REF)
	docker buildx build --platform linux/amd64 --target cuinterpose-export \
		--build-arg CUDA_HEADERS_IMAGE=$(CUDA_HEADERS_IMAGE) --output type=local,dest=bin/cuinterpose \
		-f bin/cuinterpose-src/agent/Dockerfile bin/cuinterpose-src/agent
	rm -f bin/cuinterpose/core_abi.h
	cp bin/cuinterpose-src/LICENSE bin/cuinterpose/LICENSE

# The all-in-one image a Kubernetes pod restores in (docs/kubernetes.md):
# the pinned vLLM image with warmctl, cuInterpose and tini.
pod-image:
	test -f bin/cuinterpose/libcuinterpose.so || $(MAKE) cuinterpose
	docker build -t $(POD_IMAGE) -f deploy/kubernetes/Dockerfile --build-context cuinterpose=bin/cuinterpose \
		--build-arg VERSION=$(VERSION) .

# The release archives as .github/workflows/release.yml builds them, as a
# local snapshot in dist/ (nothing signed or published). Needs goreleaser v2.
release:
	test -f bin/cuinterpose/libcuinterpose.so || $(MAKE) cuinterpose
	goreleaser release --snapshot --clean --skip=sign,sbom

# Record docs/assets/up.gif from docs/demo/up.tape (a replay of a real
# restore log) with charmbracelet/vhs in Docker.
demo:
	docker run --rm -v $(CURDIR):/vhs -w /vhs ghcr.io/charmbracelet/vhs:v0.10.0 docs/demo/up.tape

# Publish catalog/catalog.yaml to the configured registry (WARMCTL_REGISTRY
# or /etc/warmctl/config.yaml); needs push credentials (docker login).
catalog-push:
	$(GO) run ./cmd/warmctl catalog --publish catalog/catalog.yaml

clean:
	rm -rf bin dist
