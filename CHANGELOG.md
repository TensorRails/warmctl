# Changelog

## Unreleased

Restore

- The weights reload is pipelined (#38): each thread owns two page-locked
  64 MiB slots and a stream of its own, and reads the next chunk of a part
  file while the previous one is copied to the GPU (`cudaMemcpyAsync`).
  `up --sync-weights` keeps the synchronous copies.
- The reload line splits the slowest rank's time: `weights reloaded and
  fingerprints match: 122.53 GiB, slowest rank 3.7s (rank 0: copy 3.0s at
  5.1 GiB/s, pipelined, 4 threads spent read 10.7s and copy wait 0.0s;
  fingerprint 0.8s)`. On Qwen3-8B DP8 on 8×RTX PRO 6000 (15.3 GiB per rank,
  warm page cache) the pipeline took the copy wait from 0.8–1.0 s to 0–0.1 s
  and the copy from 3.4 s to 3.1 s on average; the rest is the threads'
  reads from the page cache into the slots, bound by host memory bandwidth
  with eight ranks reading at once: 8 threads per rank made the copy 3.8 s
  and 16 threads 6.9 s, so 4 stays the default.

## v2.0.0-beta.13 — 2026-10-25

Distribution

- `warmctl up NAME` restores while it pulls. The CRIU images and the runtime
  trees come first, and CRIU and the CUDA restore run on them while the
  weights still download; the engine wakes once the weights have arrived,
  reading them from memory while the disk still syncs them. The engine image
  is pulled at the same time on a host that lacks it. Every file is still
  checked against its SHA-256 before the restore reads it, and a pull that
  fails leaves no snapshot behind (#34).
- A pull syncs the disk once, after every blob is written, instead of after
  each blob: on ext4 every sync waited for all the others. From an empty
  store on a network disk, `up` reached its first token 17–21% sooner for
  Qwen3-8B (1×H200, 1×L40S), 12% for Qwen3-32B TP2 (2×L40S) and 7% for
  gpt-oss-120b (1×H200); on storage faster than the network the process
  restore runs inside the download (#36).

## v2.0.0-beta.12 — 2026-10-24

Kubernetes

- `warmctl up --tls-cert FILE --tls-key FILE` serves the gateway over HTTPS,
  from the first probe of an in-place restore on. The pair is read again
  when a file changes, so a rotated Kubernetes secret is picked up without a
  restart. KServe with `enableLLMInferenceServiceTLS` probes, scrapes and
  routes over HTTPS; the preset passes its certificate (#32, #33).
- `deploy/kubernetes/kserve.yaml`: warmctl as a KServe
  `LLMInferenceServiceConfig` preset.

## v2.0.0-beta.11 — 2026-10-24

Kubernetes

- In place, `--listen` accepts the pod's own IP on port 8000, the restored
  engine's port: KServe's LLMInferenceService sends its probes, InferencePool
  and workload Service there, so its pod runs `warmctl up --listen
  "${POD_IP}:8000"`. Port 8000 on every address or on loopback is still
  refused: it would take the engine's socket (#29).

Profiles

- `deepseek-v41-flash-1017-b300-dep8` captures graphs up to 8192 tokens
  (`extra_capture_sizes` from 1664 in steps of 128, warmed by
  `prefill_capture_sizes`), as the recipe's `max-cudagraph-capture-size 8190`
  does. Under load most steps mix decode with a prefill chunk; in DP every rank
  runs the size of the largest one, so with graphs only up to 1536 tokens
  those steps ran eagerly. Restored on 8×B300 at 1K/1K, 2048 concurrent:
  55K output tokens/s (was 37K), TPOT 29 ms (was 43 ms). The snapshot grows to
  851 GiB and restores in 88 s from the page cache.
- The hardware table has a `b300ac-275gb` row (NVIDIA B300 SXM6 AC); the
  B300 DEP8 profile accepts both B300 rows.

## v2.0.0-beta.10 — 2026-10-21

Engine

- Data-parallel profiles build and restore on vLLM's DP frontend, which runs a
  `/collective_rpc` call on every engine replica but answers with the first
  replica's workers only. With `dp > 1` warmctl calls its worker methods
  through the adapter's `warmctl_rpc`, which publishes each rank's result as a
  file in the agent's IO directory, and requires one file for each rank
  (#23). TP-only profiles and their snapshots keep the plain RPC.
- Engines run with `NCCL_GRAPH_REGISTER=0`: NCCL's registration of the user
  buffers of graph-captured collectives goes through cuInterpose's memory
  adapter, and on the r615 user-mode driver the first replay of a larger
  decode graph hung every rank (#27). Without the registration the DEP8
  engine runs, checkpoints and restores; the registration only matters for
  large buffers.

Profiles

- `deepseek-v41-flash-1017-b200-dep8`: the DEP8 recipe on 8×B200 (100 GiB of
  FP8 KV per GPU; 46 GiB of weights per rank). In the catalog for driver
  615: 809 GiB, restores in 84 s from the page cache, 38K output tokens/s at
  1K/1K once restored.
- `deepseek-v41-flash-1017-b300-tp8`: DeepSeek-V4.1-Flash-1017 on 8×B300, one
  TP8 engine with expert parallelism, 256 sequences, 16K prefill chunks, 128 GiB
  of FP8 KV per GPU and DSpark with 5 draft tokens.
- `deepseek-v41-flash-1017-b300-dep8`: the same model for throughput, after
  vLLM's B300 recipe: DEP8 (eight data-parallel attention replicas, EP with
  weight filtering), FlashMLA mega-attention with an MXFP4 sparse indexer,
  DeepGEMM mega-MoE, DSpark with block rejection sampling. Its image is the
  recipe's (vLLM 0.31) with libcuda 615 in place of 580
  (`registry.warmctl.io/warmctl/vllm-openai:0.31-dsv41-b300-cu13compat615`,
  `deploy/images/vllm-cu13compat615`): an r580 libcuda cannot restore the
  mega-MoE multicast buffers after cuda-checkpoint.

Docs

- `docs/interpose.md`: multicast objects restore only on a user-mode driver
  of r595 or newer (ai-dynamo/snapshot#510); the image's forward-compatible
  `libcuda` counts, the host kernel driver may stay on 580.
- The hardware table has a `b300-275gb` row (NVIDIA B300 SXM6, sm_103).

## v2.0.0-beta.9 — 2026-10-19

`warmctl hardware` shows which GPUs snapshots support, and `warmctl doctor`
checks that the host's GPU is one of them before a build fails on it.

CLI

- `warmctl hardware [--dir DIR]` lists the GPU models snapshots support: the
  shipped hardware table plus a `hardware.yaml` in each `--dir`, with id,
  nvidia-smi name, memory, arch, NVLink and topology, and marks the row of
  this host's GPUs.
- `warmctl doctor` has a `hardware` check: it names the host's hardware row,
  or warns with the GPU's nvidia-smi name when the table has none, before a
  `build` fails on it. `--profile-dir` adds the rows of a `hardware.yaml`
  there, as it does for `build`.

## v2.0.0-beta.8 — 2026-10-19

The NVIDIA RTX PRO 6000 Blackwell Server Edition joins the hardware table, and
Qwen3.8-27B gets a production profile and a catalog image for it.

Hardware

- `rtx-pro-6000-96gb`: RTX PRO 6000 Blackwell Server Edition, 96 GB, sm_120,
  PCIe. A `hardware.yaml` in a profile directory may now repeat a shipped row
  word for word, so a row written for an older release keeps working after a
  release ships it; a different row under a shipped id is still an error.

Profiles

- `qwen38-27b-rtx6000`: Qwen3.8-27B FP8 on one RTX PRO 6000 with 128K context,
  64 sequences, 48 GiB of KV cache and CUDA graphs up to batch 64. A vLLM cold
  start takes 215 s to engine ready; `warmctl up` from a warm page cache serves
  in 11 s. Published under `qwen38-27b` next to the B200 and H200 images.

## v2.0.0-beta.7 — 2026-10-17

Three models join the catalog: Ling-3.1-flash on 8×B200, DeepSeek-V4.1-Flash-1017
on 4×B200 and Nemotron 3.5 Super VL on 4×H100 and 2×B200. The profiles ship in
the binary, so `warmctl up NAME` and `warmctl build --profile ID` know them.

Profiles

- `deepseek-v41-flash-1017-b200-tp4`: DeepSeek-V4.1-Flash-1017, the 2026-10-17
  update of V4.1-Flash, on 4×B200. Its config is byte-identical to V4.1-Flash,
  so the profile is `deepseek-v41-flash-b200-tp4` with the new revision and the
  model card's sampling (top_p 1.0). A vLLM cold start takes 1,196 s to engine
  ready; `warmctl up` from a warm page cache serves in 78 s. Published as
  `deepseek-v41-flash-1017`, a repository of its own next to
  `deepseek-v41-flash`, and added to the catalog.
- `nemotron35-super-vl-h100-tp4`: NVIDIA Nemotron 3.5 Super VL 120B-A12B
  (BF16, hybrid Mamba MoE with vision) on 4×H100, TP4 as in NVIDIA's model
  card, MTP with 3 draft tokens, 128K context. A vLLM cold start takes 208 s
  to engine ready; `warmctl up` from a warm page cache serves in 23.6 s with a
  bitwise canary. Published as `nemotron35-super` and added to the catalog.
- `nemotron35-super-vl-b200-tp2`: the same model on 2×B200, TP2 as in the
  model card, with 16 GiB of KV cache per GPU.
- `ling31-flash-b200-tp8`: inclusionAI Ling-3.1-flash (572B MoE, FP8) on
  8×B200 with expert parallelism and MTP. FlashInfer's TRT-LLM MoE router
  accepts at most 32 experts per group and Ling has 64, so the profile sets
  `moe-backend: deep_gemm`. Both profiles pin a vLLM nightly with the
  BailingMoeV3 TP fix (vllm-project/vllm#61642) and Nemotron 3.5 Omni support.
  A vLLM cold start of Ling takes 854 s to engine ready; `warmctl up` from a
  warm page cache serves in 47 s. Published as `ling31-flash` (647 GiB) and
  added to the catalog. Benchmarks: [Benchmarking Ling-3.1-flash on 8×B200](https://warmctl.io/blog/benchmarking-ling-3-1-flash/).

Internal

- CI lints on Go 1.26 with golangci-lint v2.14.0.
- Five functions no command reaches are removed.

## v2.0.0-beta.6 — 2026-10-15

GPUs

- warmctl reads the GPUs, their compute processes and their topology from
  NVML (`libnvidia-ml.so.1`, loaded at run time) instead of running
  `nvidia-smi` three times and parsing its text. Whether GPUs are joined by
  NVLink comes from NVML's peer-to-peer status rather than the `topo -m`
  matrix; the classes (`nvlink`, `pcie`, `single`) are unchanged, so existing
  snapshots restore as before. `topology_sha256` in new manifests digests a
  matrix warmctl renders from NVML. A container needs the driver's
  `utility` capability, as it did for `nvidia-smi`.
- The agent asks the CUDA driver (`libcuda.so.1`, loaded at run time)
  whether every GPU supports VMM with POSIX handles, the check behind
  `ipc: auto`, instead of running `python3 -m warmctl_vllm.devices`; that
  module is gone, and the probe is one driver call in the agent rather than
  an interpreter start. A container needs the driver's `compute` capability
  for it, which an engine needs anyway.
- The linux binaries are now dynamically linked against glibc, which the
  NVML and CUDA loaders need; cgo stays off.

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
