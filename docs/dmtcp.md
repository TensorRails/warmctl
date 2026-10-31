# The dmtcp backend: snapshots in containers without privileges

Status: experimental (branch `exp/dmtcp-backend`), proven end to end on a RunPod A40 Pod.

GPU clouds such as RunPod Pods rent a Docker container rather than a machine
([#57](https://github.com/TensorRails/warmctl/issues/57)). CRIU cannot run in one:

- its capabilities are Docker's defaults, with no `SYS_ADMIN` or `CHECKPOINT_RESTORE`;
- seccomp refuses `unshare` and `clone3`;
- `/proc/sys` is read-only.

[DMTCP](https://github.com/TensorRails/dmtcp/tree/warmctl) checkpoints a process tree in user space.
NVIDIA's `cuda-checkpoint` works in such a container. Together they take the place of CRIU, and nothing
else in warmctl changes: weights export and reload, fingerprints, canary, trees, OCI distribution.

```bash
warmctl build --in-place --profile qwen3-0.6b-tp1 --model-dir /models/qwen3-0.6b --output /snap/q06
warmctl up /snap/q06 --mode in-place          # or up NAME: the index entry this host can restore
```

## Choosing the backend

| | Decided by |
|---|---|
| bake | `--backend auto` (the default) in place takes CRIU when the container may run it, and DMTCP otherwise. It judges from the effective capabilities, a writable `ns_last_pid` and the seccomp mode (`host.ChooseBackend`). A donor container always runs CRIU. |
| manifest | `process.backend` records the choice. Snapshots without the field are CRIU's. |
| index | `io.warmctl.backend` on each entry. A DMTCP snapshot's tag ends in `-dmtcp`, and CRIU's tags are unchanged, so one platform and driver hold both. |
| restore | `up NAME` takes the first entry `restore.CanRestore` accepts: CRIU through Docker or in a container that may checkpoint, DMTCP in place with DMTCP installed. Tags sort CRIU first. |

## How it runs

The agent's steps call one interface, `tree` (`internal/agent/tree_linux.go`). `criuTree` is the code the
agent had before, unchanged; `dmtcpTree` (`dmtcp_linux.go`) is the new backend:

| Step | criu | dmtcp |
|---|---|---|
| launch | PID allocator moved to 32768 | coordinator, engine under `dmtcp_launch --no-gzip --with-plugin libdmtcp_fdkeep.so` |
| cuda | lock + checkpoint | the same; owners recorded as DMTCP's virtual PIDs, the ones vLLM's workers report |
| freeze | SIGSTOP through pidfds | none: DMTCP stops the tree inside the dump |
| dump | `criu dump` | `dmtcp_command --bcheckpoint`, then `--kill`; images in `criu/`, so publish, layers and prefetch are shared |
| restore | `criu restore --leave-stopped` | `dmtcp_restart --no-strict-checking`, waiting until every peer runs; owners translated back to real PIDs |
| in place | tmpfs and bind mounts, UTS namespace | trees claimed, model and adapter symlinked (`run.Place`); no mount, no namespace |

The engine keeps the profile's configuration, vLLM's mp executor included (API server, engine core,
worker, resource tracker). It runs with `VLLM_WORKER_MULTIPROC_METHOD=fork`, one-thread BLAS pools,
`UV_USE_IO_URING=0` and `UCX_MEM_MMAP_HOOK_MODE=none` (`dmtcp.Env`). Tensor parallelism restarts too: the collectives are detached before
the checkpoint and set up again after the restore, as under CRIU.

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

Library text now comes from the receiver's files, so donor and receiver need the same DMTCP build and
libraries, which the runtime image gives them. A mismatch stops the restart; it never maps other code.

The plugin `internal/dmtcp/fdkeep/fdkeep.c` repairs what DMTCP loses:

- the CUDA driver's eventfds and pipes, recreated as real pipes;
- thread names, which `cuda-checkpoint` uses to find the restore thread;
- `listen` on sockets DMTCP brought back unbound;
- epoll watches lost when a forked child was elected the leader of a shared epoll fd.

## Numbers

RunPod A40 (Secure Cloud, EU-SE-1), Qwen3-0.6B, vLLM 0.32.0, profile `qwen3-0.6b-tp1`. The snapshot is
5.33 GiB: 4.4 GB of process images and 1.2 GiB of weights.

| Start | Seconds to serving |
|---|---|
| vLLM, empty compile caches | 133–155 |
| vLLM, warm compile caches | 34 |
| warmctl, snapshot on local disk, page cache evicted | 4.0–4.3 |
| warmctl, snapshot in the page cache | 3.8–4.1 |
| warmctl, snapshot on a Global Volume (S3 behind FUSE), first `up` | 9.8 (5.2 of it copying 5.33 GiB at 1.03 GiB/s) |
| warmctl, same, later `up` (the local copy is reused) | 4.4 |

Five restores in a row passed. A restore breaks down as: `dmtcp_restart` 1.7–2.1 s, CUDA 0.4–0.6 s,
weights 0.5 s at 2.3 GiB/s, canary 0.3 s.

Two A40s (RunPod, $1.18/h), Qwen3-0.6B with TP=2 (`qwen3-0.6b-a40-tp2`, mp executor: 5 processes, 4 CUDA
owners). Images 9.24 GB, snapshot 11.16 GiB.

| Start | Seconds to serving |
|---|---|
| warmctl, snapshot in the page cache | 5.0–5.3 |
| warmctl, page cache evicted | 5.6–5.7 |

Four restores in a row passed: `dmtcp_restart` 1.8–2.0 s, CUDA 0.7–0.8 s, weights 0.3 s per rank.

## Limits and open work

- **TP=2 is verified on Qwen3-0.6B only.** Larger models with several ranks, and NCCL's P2P transport,
  are not verified yet. On RunPod's 2×A40 hosts P2P all-reduces hang with or without DMTCP, though the
  driver reports peer access (ACS/IOMMU), so the TP=2 profile sets `NCCL_P2P_DISABLE=1` (SHM transport).
- A restore warns that 3 sockets cannot bind their port again (`EADDRINUSE`); serving and the canary are
  not affected. Which sockets these are is open.
- No interposed IPC (cuInterpose).
- RunPod mounts volumes at `/workspace`, which warmctl claims as a tree, and its image ships
  `/workspace/.cache`. Mount volumes elsewhere.
- A Global Volume is S3 behind geesefs (FUSE). It cannot hold an OCI layout, because oras-go `chmod`s
  blobs. It serves a plain snapshot directory, at about 124 MB/s from one stream. FUSE keeps no page cache
  the prefetch could fill for `dmtcp_restart` and the weights reload, which read one stream each: 61 s
  to serving. So `up` from a FUSE directory first copies it to `<state>/staged/<id>`, 32 ranges in
  flight (`restore.Stage`), and reuses a complete copy.
- DMTCP is built from the fork:

```bash
./configure --prefix=/opt/dmtcp && make -j && make install
gcc -shared -fPIC -O2 -I/opt/dmtcp/include -o /opt/dmtcp/lib/libdmtcp_fdkeep.so internal/dmtcp/fdkeep/fdkeep.c
```
