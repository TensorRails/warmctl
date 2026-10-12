# Changelog

## Unreleased

Releases

- `curl -fsSL https://warmctl.io/install.sh | sh` installs the newest
  release, pre-releases included, in `/opt/warmctl` and links
  `/usr/local/bin/warmctl`; run it again to upgrade. It checks the archive
  against the release checksums and, with cosign on PATH, their signature.
  `WARMCTL_VERSION`, `WARMCTL_PREFIX`, `WARMCTL_BIN_DIR` and
  `WARMCTL_DOWNLOAD` (a mirror) change what and where. The script is
  [scripts/install.sh](scripts/install.sh).
- Every release ships archives for linux/amd64 (with cuInterpose),
  linux/arm64 and macOS (amd64, arm64), built by GitHub Actions from the tag.
  The binary is `warmctl` in every archive (was `warmctl-linux-amd64`).
  linux/arm64 has no cuInterpose build yet and is not proven on a GPU; on
  macOS warmctl runs the registry side (`catalog`, `images`, `mirror`, `push`,
  `pull`, `profile`).
- `warmctl-<tag>-checksums.txt` is signed keylessly with cosign, every archive
  has an SBOM and build provenance (`gh attestation verify`); how to check
  them: [RELEASING.md](RELEASING.md#verify-a-release).
- CI also tests on arm64, cross-compiles every release platform and runs
  govulncheck; actions are pinned by commit and kept current by Dependabot.

## v2.0.0-beta.5 — 2026-10-12

The first public release. Restore in a Kubernetes pod with the same
`warmctl up NAME`, multi-GPU engines restore under NVIDIA driver 610, and a
cold restore from a file share reads the snapshot in parallel (GLM-5.3-Flash
on 8×H100: 75 s → 57 s).

Kubernetes (prototype)

- `up` picks where to restore. In the warmctl runtime image (marker
  `/opt/warmctl/RUNTIME`) it restores in place, in its own container; otherwise
  it asks Docker for a receiver container. With neither, it stops with an
  error. `--mode host|in-place` overrides the choice.
- In place, warmctl prepares its privileged container as a receiver: tmpfs at
  `/tmp` and `/dev/shm`, the snapshot's extension and cuInterpose mounted
  read-only, the GPUs the device plugin assigned. The agent runs in a UTS
  namespace of its own with the host name the engine had. The pod needs no
  volumes; a node-local `/var/lib/warmctl` only keeps pulled snapshots.
- The preflight refuses `hostNetwork` and `hostPID`, a missing capability or
  seccomp profile, a read-only `/proc/sys` and an image built from another
  vLLM base. warmctl empties `/workspace` and its run directory only when they
  are empty or marked by an earlier restore; foreign data stops the restore
  untouched.
- The gateway listens on `:8080` from the start: `/health` answers 503 while
  pulling and restoring and 200 after the canary, so a `readinessProbe` keeps
  traffic away until the engine is warm.
- The runtime image is `registry.warmctl.io/warmctl/runtime`, a tag per
  release and `:latest`; `make pod-image` builds it for another registry or
  vLLM base. `deploy/kubernetes` holds its Dockerfile, a Deployment and
  Service, a node-preparation DaemonSet and a KEDA ScaledObject on
  `vllm:num_requests_waiting`. Design and open risks:
  [docs/kubernetes.md](docs/kubernetes.md).
- Tested on CPU in pod-like containers (`make e2e-test`), not yet on a GPU
  node of a Kubernetes cluster.

Driver 610

- Multi-GPU engines under driver branch 595 and later run on the host's
  libcuda: `build` puts the host's library directory first in
  `LD_LIBRARY_PATH`, because the vLLM image's forward-compatible libcuda
  cannot checkpoint processes that share GPU memory under a 610 kernel
  module. Single-GPU engines keep the image's libcuda.
- Published for 610.57.04: eight single-GPU H100 snapshots, gpt-oss-120b on
  2×H100 and GLM-5.3-Flash on 8×H100.

Restore from a file share

- `up` reads the snapshot ahead from the start of the restore: the CRIU
  images first, then the weights round-robin across ranks, with 32 parallel
  readers capped by available memory (`WARMCTL_PREFETCH_WORKERS` overrides).
  Each range is also handed to the kernel with `posix_fadvise(WILLNEED)`, and
  `up` logs how the prefetch went. GLM-5.3-Flash on 8×H100 restored cold from
  a virtiofs share in 56.7 s instead of 75.1 s; a warm restore stays at
  40–44 s.

Known issues

- Under driver 610, MiMo-V2.6-Flash and DeepSeek-V4.1-Flash do not start on
  the host's libcuda: their kernels ship as CUDA 13.4 PTX, which needs a newer
  libcuda to compile.
- The first restore right after a host boots can fail in CRIU (the restored
  root task is killed after about a second); later restores pass.
- `warmctl pull` onto a virtiofs share writes at about 20 MB/s.
- GLM-5.3-Flash at TP4 on B200 fails in a Triton kernel of its linear
  attention under 32 users with 4K random-token prompts; TP8 and ordinary
  text work.

## Before beta.5

Earlier v2 releases were not published. What they brought:

- **v2.0.0-beta.4.** Snapshots restore under any release of their NVIDIA
  driver branch: `build` runs the engine with the driver's version check off
  and keeps the driver libraries the dump maps in the snapshot. GLM-5.3-Flash
  on 8×B200 (44 s) and 4×B200, DeepSeek-V4.1-Flash on 4×B200 (73 s).
- **v2.0.0-beta.3.** Models by name: `warmctl catalog`, `up NAME` pulling the
  snapshot for this host's platform from a registry, a mirror or a shared
  cache, `mirror`, resumable pulls, anonymous pulls from
  `registry.warmctl.io`. Commands renamed to `build`, `up`, `down`, `images`
  and `ps`, with the old names as aliases.
- **v2.0.0-beta.2.** Snapshots of any size through an OCI registry: large
  files as pieces, small ones as bundles, 64 parallel HTTP/1.1 connections,
  chunked uploads, retries. DeepSeek-V4.1-Flash on 8×H100 restores in
  75–86 s; B200 support.
- **v2.0.0-beta.1.** Cross-process CUDA memory through NVIDIA cuInterpose, so
  fast collectives (custom, FlashInfer and symmetric-memory all-reduce, NCCL
  NVLS) survive a snapshot. GLM-5.3-Flash prod TP8 restores in 37 s at stock
  serving speed.
- **v2.0.0-alpha.1.** The v2 rewrite: snapshot a warmed vLLM engine and
  restore it in seconds, profiles in YAML only, OCI distribution.
