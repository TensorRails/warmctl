# Cross-process CUDA memory

cuda-checkpoint checkpoints one process at a time and refuses memory another
process maps: legacy IPC (`cudaIpcGetMemHandle`), VMM exported as POSIX file
descriptors, multicast objects. vLLM's fast collectives (custom all-reduce,
symmetric memory, FlashInfer, NCCL NVLS) and model-private workspaces
(MiniMax, DeepSeek-V4.1, …) all live there. warmctl solves this once, below
vLLM, with NVIDIA's cuInterpose:

- `libcuinterpose.so` is preloaded into the engine. It records every
  cross-process object when it is created; `cudaMalloc` becomes a tracked VMM
  allocation, so legacy IPC can be re-created later.
- Before the CUDA checkpoint, `cuinterpose-coordinator --prepare` saves the
  bytes of every shared allocation to host memory, which CRIU dumps, and
  releases the sharing while keeping every virtual address reserved. Its
  record goes into the snapshot as `ipc/cuinterpose.state`.
- After the CUDA restore, `--restore` re-creates the allocations, imports and
  multicast objects at the same addresses and loads the bytes back. CUDA
  graphs, NCCL communicators and Lamport flags come back unchanged.

Nothing in warmctl knows which library shared what. A snapshot that holds
cross-process memory without the interposer fails at the CUDA checkpoint.

## When it is used

The profile's `ipc` decides:

| `ipc` | Engine | Use |
|---|---|---|
| `auto` (default) | `interpose` when the engine has more than one GPU process and every GPU supports VMM with POSIX handles, else `native` | every profile |
| `interpose` | under cuInterpose; the bake fails if a GPU lacks VMM with POSIX handles | profiles whose fast collectives must not fall back |
| `native` | stock: no preload | opting out |

Only an engine with more than one GPU process can share memory, so a TP1
engine never changes. The VMM check is a driver attribute read inside the
donor before the engine starts (`cuDeviceGetAttribute` through
`libcuda.so.1`, which the agent loads at run time), not a GPU name: an L40S TP2 engine, whose custom all-reduce shares memory over
PCIe, is interposed like an H100 one; a vGPU without VMM runs native. The
choice is recorded in the manifest and the restore follows it.

## Packaging

`make cuinterpose` builds the pinned release in Docker (`CUINTERPOSE_REF`,
CUDA headers from Docker Hub). `make release` puts the four files next to the
binary under `cuinterpose/`. A bake copies them into the snapshot's adapter
tree, so a restore runs the exact libraries the dumped processes have loaded
and the coordinator that wrote their state.

## Steps

Snapshot: sleep → `checkpoint_prepare` → `ipc` (`--prepare`) → `cuda` →
freeze → dump → fs. Restore: install-fs → criu → thaw → `cuda` → `ipc`
(`--restore`) → wake. After `--prepare` no process maps another's memory,
so every CUDA owner restores in parallel.

Only processes with a cuInterpose control socket take part. A process opens
it on its first intercepted CUDA call; one that only loaded the driver (an
API server) holds no shared memory and takes the plain CUDA checkpoint.

The libraries are mounted read-only and with the store's file mode in donor
and receiver: CRIU maps a library back only onto a file of the same path,
size and mode.

## Cost

The interposer only sees allocation and mapping calls, never kernel
launches. `cudaMalloc` through VMM is slower to allocate and rounds up to the
2 MiB granularity; PyTorch's caching allocator keeps that out of steady
state. Measured on GLM-5.3-Flash prod TP8 against stock vLLM with the same
flags: throughput and inter-token latency equal, restore 37 s. Preparing and
restoring the shared memory takes 0.1–1.1 s.
