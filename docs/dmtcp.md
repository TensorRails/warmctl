# The dmtcp backend: snapshots in containers without privileges

Status: experimental. Verified end to end on RunPod Pods (A40 and L40S, Secure Cloud), and on a Nebius
RTX PRO 6000 host in Docker containers given RunPod's privileges (none). This covers:

- one GPU, the mp executor and TP=2;
- restores on another pod and another host;
- snapshots on a Global Volume and on a network volume.

GPU clouds such as RunPod Pods rent a Docker container rather than a machine
([#57](https://github.com/TensorRails/warmctl/issues/57)). CRIU cannot run in one:

- its capabilities are Docker's defaults, with no `SYS_ADMIN` or `CHECKPOINT_RESTORE`;
- seccomp refuses `unshare` and `clone3`;
- `/proc/sys` is read-only.

[DMTCP](https://github.com/TensorRails/dmtcp/tree/warmctl) checkpoints a process tree in user space.
NVIDIA's `cuda-checkpoint` works in such a container. Together they take the place of CRIU, and nothing
else in warmctl changes: weights export and reload, fingerprints, canary, trees, OCI distribution.

```bash
# In a pod running the warmctl runtime image:
warmctl build --in-place --profile qwen3-8b-a40-tp1 --model-dir /nv/models/qwen3-8b --output /snap/q8 --compress
rsync -a --exclude manifest.json /snap/q8/ /nv/snap/q8/ && cp /snap/q8/manifest.json /nv/snap/q8/
# In any other pod of the same image, GPU model and driver:
warmctl up /nv/snap/q8 --mode in-place        # or up NAME: the index entry this host can restore
```

## Choosing the backend

| | Decided by |
|---|---|
| bake | `--backend auto` (the default): DMTCP in place (`--in-place`), CRIU in the donor container warmctl makes. An in-place CRIU bake is refused as not supported yet: the bake links the model in, so CRIU would record files under the link's target, which an in-place CRIU restore does not bring back. |
| manifest | `process.backend` records the choice. Snapshots without the field are CRIU's. |
| index | `io.warmctl.backend` on each entry. A DMTCP snapshot's tag ends in `-dmtcp`, and CRIU's tags are unchanged, so one platform and driver hold both. |
| restore | `up NAME` takes the first entry `restore.CanRestore` accepts: CRIU through Docker or in a container that may checkpoint, DMTCP in place with DMTCP installed. Tags sort CRIU first. |

## How it runs

The agent's steps call one interface, `tree` (`internal/agent/tree_linux.go`). `criuTree` is the code the
agent had before, unchanged; `dmtcpTree` (`dmtcp_linux.go`) is the new backend:

| Step | criu | dmtcp |
|---|---|---|
| launch | PID allocator moved to 32768 | a coordinator of the run's own, on a free loopback port; engine under `dmtcp_launch --no-gzip --with-plugin libdmtcp_fdkeep.so` |
| cuda | lock + checkpoint | the same; owners recorded as DMTCP's virtual PIDs, the ones vLLM's workers report |
| freeze | SIGSTOP through pidfds | none: DMTCP stops the tree inside the dump |
| dump | `criu dump` | `dmtcp_command --bcheckpoint`, then `--kill`; images in `criu/`, so publish, layers and prefetch are shared |
| restore | `criu restore --leave-stopped` | a new coordinator, `dmtcp_restart --no-strict-checking`, waiting until every peer runs; owners translated back to real PIDs. A peer that dies, or `dmtcp_restart` exiting, fails the restore at once, killing every restarted process and the coordinator. |
| in place | tmpfs and bind mounts, UTS namespace | trees claimed, model and adapter symlinked (`run.Place`); no mount, no namespace |

Each run starts its own coordinator on a free port and ends it when the agent exits, so a coordinator
left by a failed run (one no process joined waits forever) never blocks the next. A container still runs
one in-place run at a time: the engine owns its `/tmp`, `/dev/shm` and `/workspace`. A second `up` or
`build --in-place` is refused while the first runs (it holds a flock on the container's `/`, which a
read-only root allows); for two engines on a pod's two GPUs, run two containers.

The engine keeps the profile's configuration, vLLM's mp executor included (API server, engine core,
worker, resource tracker). It runs with the variables in `dmtcp.Env`:

| Variable | Why |
|---|---|
| `VLLM_WORKER_MULTIPROC_METHOD=fork` | With spawn, the engine core inherits a pipe that DMTCP turned into a socketpair and never drains it, so the checkpoint never ends. |
| `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1` | A fork while OpenBLAS or OpenMP thread pools are running hangs the child. |
| `UV_USE_IO_URING=0` | libuv must not batch `epoll_ctl` through io_uring, which DMTCP cannot see. |
| `UCX_MEM_MMAP_HOOK_MODE=none` | UCX, which the vLLM image ships for NIXL, hooks `mmap` and `shmat` and calls the originals past DMTCP's wrappers. |

Tensor parallelism restarts too: the collectives are detached before the checkpoint and set up again after
the restore, as under CRIU. Collectives that share GPU memory between processes through CUDA IPC (FlashInfer's
PCIe all-reduce, custom all-reduce) cannot be detached and keep their addresses in CUDA graphs; with `ipc: auto`
the engine runs under cuInterpose, which takes the sharing apart before the checkpoint and puts it back at the
same addresses after the restore ([interpose.md](interpose.md)). Under DMTCP:

- The engine maps the runtime image's own build, `/usr/local/bin/cuinterpose`. Nothing is mounted for a DMTCP
  engine, and a restart maps a library again from its file only when its size and mtime match, so donor and
  receiver both use the image's copy. The manifest records the path (`process.ipc_build`), and a restore checks
  that the libraries there are the snapshot's.
- A process names its control socket after the PID it sees, the virtual one, and so does the coordinator.
- cuInterpose makes the socket with raw syscalls, which DMTCP never sees, so a restart would lose it. fdkeep
  records every listening Unix socket bound to a path and binds a missing one again at its descriptor, with
  its flags and the path's permissions.
- The coordinator's state is read from the snapshot's `ipc/`.

### Another pod

A pod restores a snapshot another pod took when it runs the same image, GPU model and driver.

- **GPU device nodes.** Hosts number `/dev/nvidiaN` in their own order: the same rank was `/dev/nvidia3`, `1`,
  `5` and `0` on four RunPod pods. DMTCP reopens a descriptor by its path. So the manifest records each GPU's
  node (`hardware.gpus[].device`, from the driver's Device Minor), and a restore links a node this pod lacks
  to its own GPU of the same rank. CUDA itself moves by UUID (the device map).
- **The model's path.** An in-place bake links the model in, so the processes hold its files under its real
  directory, which the manifest records (`launch.model_dir`, symbolic links resolved). A restart reopens
  them there: the receiving pod must have the model at that same path, as the same volume mounted at the
  same place gives it. `up` checks that before anything else, and refuses a `--model-dir` that names
  another directory.
- **The engine's image.** An in-place bake records the runtime image's base (`/opt/warmctl/RUNTIME`) as the
  engine's image, which a receiver compares with its own.
- **DMTCP.** Library text is mapped from the receiver's files (see the fork below), so DMTCP and every library
  must be byte-identical: the same image. `--dmtcp-dir` defaults to the image's `/opt/warmctl/dmtcp/bin`, not
  PATH. A shell in a RunPod pod lacks the image's PATH, and any other DMTCP would not match.

## Restoring from network storage

A RunPod Pod gets its snapshot from a Global Volume (S3 behind geesefs, FUSE on the host) or a network
volume (MooseFS over FUSE in EU-NL-1, VAST NFS in CA-MTL-1). The restore's readers, `dmtcp_restart` and
the weights reload, read one stream each, and FUSE keeps no page cache the prefetch could fill for them.

So `up` from a FUSE directory, or from any snapshot stored encoded, stages a local copy (`restore.Stage`):

- **Many reads at once.** 32 workers, 64 MiB ranges or one frame each. One reader gets 0.11 GiB/s from a
  Global Volume; 16 or more get its ceiling of about 0.5 GiB/s per pod.
- **Restore while copying.** The copy is the restore's `Arrival`, as a registry pull is. The files go lane by
  lane: the receiver's launch files, the process images and trees, then the weights. `dmtcp_restart` and the
  CUDA restore run while the weights still come.
- **Weights reloaded as they land.** A staged file takes its own name only once complete, as a pull links a
  blob in. The adapter waits for each part's file before it reads it. A snapshot whose adapter does that
  (`launch.reload_waits`, set by this bake) wakes the engine right after the process restore, and a failed
  arrival ends the reload. The adapter has no deadline of its own, as the last part comes only after all the
  weights; the stage fails when no range has been copied for 5 minutes, a read that hangs. Older snapshots
  wait for all the weights first.
- **Late files.** A Global Volume is eventually consistent: a pod saw another pod's `manifest.json` before the
  weights next to it. The stage waits up to two minutes for every file the manifest lists, at its stored size.
  Writers should copy `manifest.json` last.
- **Reuse.** A complete copy (its `manifest.json` matches) is reused by later `up`s on the pod.

### `build --compress`

The process images and runtime trees compress well, and the weights compress somewhat. `build --compress`
stores them in the output directory as independent zstd frames of 32 MiB (`internal/zframes`). The manifest
records each file's encoding, frame ends and byte-plane split; sizes and SHA-256 stay the file's own.

- **Process images: 2.84×.** Qwen3-8B's four images go from 7.28 to 2.56 GB.
- **BF16 weights as two byte planes: 1.46×.** The sign and exponent sit in the high byte, which zstd takes to
  0.37; the mantissa in the low byte does not compress. zstd on the weights as they are gets 1.28×. The bake
  picks planes when the first frame says they store less.
- **Decoding.** The stage reads and decodes every frame on its own worker, and writes the raw file. On a
  9-vCPU pod it decodes 7 GiB/s of images and 3.1–3.4 GiB/s of weights with 8 workers, more than any volume
  delivers.
- **After the bake.** `warmctl compress SRC DST` writes a compressed copy of a snapshot built without
  `--compress`. The source keeps its files, so one bake gives both: a snapshot `push` publishes, and a smaller
  copy for a network volume.

Compression stays out of the restore itself. The fork can pipe its images through zstd (`DMTCP_ZSTD=1`,
commit `fbbaf3f`, after the one warmctl pins), but `mtcp_restart` then reads one decoded stream through a
pipe at 0.67 GB/s. A restore from a local copy went from 10.3 to 15.4 s that way, so the images stay raw on
disk.

`push` refuses an encoded directory: blobs are named by the files' own digests. Compressed registry blobs
are [#60](https://github.com/TensorRails/warmctl/issues/60).

## DMTCP: the fork and the plugin

Branch `warmctl` of [TensorRails/dmtcp](https://github.com/TensorRails/dmtcp/tree/warmctl) is DMTCP 5.0
with these patches:

| Patch | Why |
|---|---|
| Map unmodified file areas from their file | A checkpoint wrote every library's text in full. A clean, private, unwritable mapping now gets a header only, and `mtcp_restart` maps the file again after checking its size and mtime. vLLM's images went from 12.3 to 4.4 GB. |
| Restore `SO_REUSEADDR` before bind | A restart right after the donor died failed to bind its port while connections sat in TIME_WAIT. |
| Find `_files` copies next to the moved image | Images published into the store restarted only if the donor's directory still existed. |
| Name a forked child's connections with its own UniquePid | A forked child named its connections `0-0-0(n)` until its first checkpoint, so vLLM's engine core and worker gave different sockets the same name, and the restart waited forever at `Socket::Restart_Ns_Send_Queries`. With the fix, the mp executor restarts. |
| Open the IPv4 restore listener for IPv6 connections | IPv6 connections are rewired through the IPv4 listener, which opened only in processes with an IPv4 socket. A TP worker whose sockets were all IPv6 (gloo, the TCPStore on `[::]`) published a zeroed address and its peer's `connect` failed. |
| Reconnect a non-blocking socket blocking | A socket created with `SOCK_NONBLOCK` came back non-blocking, and the rewirer's `connect` returned `EINPROGRESS`. |
| Pass through SysV shm ids DMTCP did not issue | UCX's memory hooks (in the vLLM image, for NIXL) create a test segment through `dlsym(RTLD_NEXT)`, past DMTCP's `shmget`, then remove it with the wrapped `shmctl`, which asserted: the engine core died at startup. |

The plugin `internal/dmtcp/fdkeep/fdkeep.c` repairs what DMTCP loses:

- the CUDA driver's eventfds and pipes, recreated as real pipes;
- thread names, which `cuda-checkpoint` uses to find the restore thread;
- `listen` on sockets DMTCP brought back unbound;
- epoll watches lost when a forked child was elected the leader of a shared epoll fd.

## The runtime image

`deploy/kubernetes/Dockerfile` builds the fork at a pinned commit and fdkeep against it, on the base, into
`/opt/warmctl/dmtcp`.

- **Base.** The default base (vLLM's cu134 nightly, CUDA 13.4) runs on RunPod's 595.91 driver (CUDA 13.2)
  through CUDA's minor-version compatibility: verified on 4×RTX PRO 6000 in EU-RO-1. The experimental image
  below was built on `vllm/vllm-openai:v0.32.0` before that was checked.
- **Published.** An experimental build, warmctl at `e5aebe7` (before `--compress`), is
  `registry.warmctl.io/warmctl/runtime:exp-dmtcp-e5aebe7` (`sha256:fc71c9bb…`).
- **Pushing it.** registry.warmctl.io sits behind Cloudflare, which refuses request bodies over 100 MB. A base
  copied into it needs chunked uploads (`regctl registry set registry.warmctl.io --blob-chunk 52428800
  --blob-max 52428800`).

## Numbers

**RunPod A40 (EU-SE-1), Qwen3-0.6B, profile `qwen3-0.6b-tp1`, snapshot on local disk.** The snapshot is 5.33
GiB: 4.4 GB of process images and 1.2 GiB of weights.

| Start | Seconds to serving |
|---|---|
| vLLM, empty compile caches | 133–155 |
| vLLM, warm compile caches | 34 |
| warmctl, page cache evicted | 4.0–4.3 |
| warmctl, snapshot in the page cache | 3.8–4.1 |

A restore breaks down as: `dmtcp_restart` 1.7–2.1 s, CUDA 0.4–0.6 s, weights 0.5 s at 2.3 GiB/s, canary 0.3 s.

**Two A40s, Qwen3-0.6B with TP=2 (`qwen3-0.6b-a40-tp2`).** The mp executor runs 5 processes with 4 CUDA
owners, and the images are 9.24 GB. warmctl reaches serving in 5.0–5.3 s from the page cache and 5.6–5.7 s
with it evicted, 4 of 4 runs.

**Qwen3-8B on a new pod: model, compile caches and snapshot all on the volume.** That is the case of a replica
added to a running service.

| Start | Global Volume (A40) | Network volume, EU-NL-1 (L40S) |
|---|---|---|
| vLLM, empty compile caches | 162 s | 124 s |
| vLLM, compile caches on the volume | **125 s** | **83 s** |
| warmctl, copy, then restore | ~58 s | 35.9 s |
| warmctl, restore while copying | — | 28–30 s |
| warmctl, `--compress` (images), restore while copying | — | 24.9 s |
| warmctl, `--compress`, weights reloaded as they land | 48.9 s (volume at 0.32 GiB/s; weights not overlapped) | **20.2–22.9 s** (volume at 0.64–0.73 GiB/s) |
| warmctl, the pod's local copy (later `up`s) | 11.4 s | 11.1 s |

The compressed 8B snapshot is 13.35 GiB instead of 23.11. Qwen3-0.6B compresses from 7.71 to 2.54 GiB and
serves 7.9–8.9 s after a first `up` from the network volume (4.9 s from the local copy).

**Storage ceilings per pod, cold data, 8 MiB reads:**

| Storage | 1 reader | Best |
|---|---|---|
| Global Volume, same region | 0.11 GiB/s | 0.50 GiB/s at 16+ readers |
| Global Volume, another region | 0.11 GiB/s | 0.36 GiB/s |
| Network volume, EU-NL-1 (MooseFS, STANDARD) | 0.28 GiB/s | 0.8 GiB/s at 4+ readers; 0.64–1.03 GiB/s in `Stage` |

The network volume's throughput varies from run to run. A Global Volume read right after another pod wrote it
runs at that pod's upload speed (0.22 GiB/s): geesefs uploads in the background after `cp` returns.

## Limits and open work

- **TP=2 is verified on Qwen3-0.6B (2×A40) and Qwen3-32B (2×L40S, the catalog profile `qwen3-32b-l40s-tp2`).**
  The 32B bake took 4 min 24 s (DMTCP dump of 5 processes in 3.1 s, 72.6 GiB); two `up`s from the pod's disk,
  page cache warm, served in 14.2 s each: restart 2.2–2.3 s, CUDA 0.7–0.8 s, 61.2 GiB of weights at 5.1–5.2
  GiB/s per rank, fingerprints and canary matching. The engine took 105 s to start cold. Restores on another
  pod are verified at TP=2 for 0.6B only; NCCL's P2P transport is not verified under DMTCP.
- **A peer path the driver reports but the host does not carry.** On RunPod's 2×L40S hosts, as on its 2×A40
  hosts, `cuDeviceCanAccessPeer` says yes and enabling peer access succeeds, but a 256 MiB peer copy does not
  finish in 20 s (21.9 GB/s through host memory), and a 64 MiB one returned after a minute with other bytes. `NCCL_P2P_DISABLE=1` reaches only NCCL: vLLM's custom
  all-reduce trusts the driver and maps the other rank's memory through CUDA IPC. Qwen3.8-27B-FP8 and Qwen3-32B
  at TP=2 then stalled at warmup (both GPUs at 100%), generated 0.1–0.2 tokens/s, or failed with
  `CUBLAS_STATUS_INTERNAL_ERROR` and left both GPUs in `ERR!` until the pod was replaced. This is not DMTCP's:
  without it the engine crawls the same way. `build` now copies between the GPUs of a PCIe host before the engine
  starts (`host.ProbePeers`, in a child process with a 10 s limit), and where a copy does not finish it turns
  custom all-reduce, FlashInfer's and symmetric-memory all-reduce and NCCL P2P off for that engine and logs why
  (`snapshot.PeerOff`). Hosts whose peer path works keep the profile's collectives. `up` does not probe: a
  snapshot baked with the peer path on, restored on such a host, still hangs.
- **CUDA IPC without cuInterpose fails the checkpoint.** A TEP4 engine with FlashInfer's PCIe all-reduce
  (`VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1`) on 4×RTX PRO 6000 failed at `cuCheckpointProcessCheckpoint` with
  error 304: cuda-checkpoint refuses a process holding another's memory. Hence cuInterpose under DMTCP, above.
- **The checkpoint signal is DMTCP's default, SIGUSR2.**
  - vLLM installs a SIGUSR2 handler, which DMTCP refuses with a warning per process; serving is not affected.
  - Do not set `DMTCP_SIGCKPT` or `--ckpt-signal`. vLLM's, CUDA's, NCCL's and gloo's threads block every
    signal, and the fork's `pthread_sigmask` wrapper unblocks only SIGUSR2. With another signal the dump never
    ends (the processes stay PRESUSPEND).
- A restore warns that 3 sockets cannot bind their port again (`EADDRINUSE`); serving and the canary are not
  affected. Which sockets these are is open.
- RunPod mounts volumes at `/workspace`, which warmctl claims as a tree, and its image ships
  `/workspace/.cache`. Mount volumes elsewhere.
- A Global Volume cannot hold an OCI layout: oras-go `chmod`s blobs and geesefs has no permission bits. It
  serves a plain snapshot directory.
- Not measured yet: a HIGH_PERFORMANCE network volume (VAST NFS, 5.7 GB/s measured earlier), and the Global
  Volume with the weights overlapped.
- Once, a repeat `up` failed at the CUDA step with `cuCheckpointProcessGetState: initialization error (3)`; it
  did not recur.
- DMTCP built by hand from the fork:

```bash
./configure --prefix=/opt/warmctl/dmtcp && make -j && make install
gcc -shared -fPIC -O2 -I/opt/warmctl/dmtcp/include -o /opt/warmctl/dmtcp/lib/libdmtcp_fdkeep.so \
  internal/dmtcp/fdkeep/fdkeep.c
```
