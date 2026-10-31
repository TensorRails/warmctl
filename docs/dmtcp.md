# The dmtcp backend: snapshots in containers without privileges

Status: experimental (branch `exp/dmtcp-backend`). The backend is meant for GPU clouds that rent a Docker
container instead of a machine, such as RunPod Pods. There CRIU cannot run at all
([#57](https://github.com/TensorRails/warmctl/issues/57)):

- the effective capabilities are Docker's defaults, with no `SYS_ADMIN` and no `CHECKPOINT_RESTORE`;
- seccomp refuses `unshare` and `clone3`;
- `/proc/sys` is read-only.

DMTCP checkpoints a process tree entirely in user space. NVIDIA's `cuda-checkpoint` works in such a
container, and it moves the CUDA state exactly as it does under CRIU.

## What changes, and what does not

`build --in-place` bakes in the current container: the agent runs as a child of `warmctl`, and the
engine runs as a child of the agent. There is no Docker. `--backend auto` picks CRIU when the container
may run it, and DMTCP when it may not. It decides from the effective capabilities, a writable
`ns_last_pid` and the seccomp mode (`host.ChooseBackend`). The manifest records the choice in
`process.backend`. `up --mode in-place` reads it from there: a dmtcp snapshot restarts under DMTCP and
asks for no privileges.

| Step | criu | dmtcp |
|---|---|---|
| start | PID allocator moved to 32768, engine started | coordinator started; engine under `dmtcp_launch --no-gzip --with-plugin libdmtcp_fdkeep.so` |
| warm up, export, sleep, canary | unchanged | unchanged |
| cuda | lock + checkpoint per owner | the same; owners reported as DMTCP's virtual PIDs |
| freeze | SIGSTOP through pidfds | none (DMTCP stops the tree inside the dump); the tree is reported in virtual PIDs |
| dump | `criu dump` | `dmtcp_command --bcheckpoint`, then `--kill`; images land in `criu/`, so publish, OCI layers and the restore's prefetch are unchanged |
| install-fs | trees into tmpfs (mounted) | trees into claimed directories (no mounts) |
| criu | `criu restore --leave-stopped` | `dmtcp_restart` on every image; waits until every peer runs |
| thaw | SIGCONT | none |
| cuda | restore + unlock | the same, after virtual-to-real PID translation (`dmtcp_command --list`) |
| mounts in place | bind mounts of the adapter and the model; UTS namespace | symlinks (`run.Place`); no namespace |

Paths must match on both sides, because DMTCP reopens files by their real path. In place, the model,
the state directory and the snapshot therefore live at the same paths on the donor and the receiver: a
network volume mounted at the same place on every pod.

## What DMTCP needed

Each problem below was found on a RunPod A40 with `vllm serve` (API server and engine core in separate
processes) and vLLM 0.32.0:

| Symptom | Cause | Fix |
|---|---|---|
| `cuda-checkpoint`: "Could not find restore thread" | DMTCP drops the eventfds and pipes `cuInit` makes, renames threads `DMTCP:<name>`, and restores pipes as socketpairs (`/proc/PID/fd/N` reopen gives `ENXIO`) | fdkeep recreates the missing eventfds and pipes (real pipes, raw syscall) and restores thread names, in `DMTCP_EVENT_THREAD_RESUME` |
| `import torch` hangs under `dmtcp_launch` | a fork (ldconfig for `find_library`) while 63 OpenBLAS threads live | `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` (`dmtcp.Env`) |
| `--bcheckpoint` never returns, API server in `SocketConnList::drain` | `spawn` hands the engine core `os.dup(pipe_handle)` of a DMTCP socketpair; the child's DMTCP does not drain it | `VLLM_WORKER_MULTIPROC_METHOD=fork` |
| `--bcheckpoint` never returns with `DMTCP_SIGCKPT=40` | ZMQ's background threads block every signal; DMTCP's sigmask wrappers unblock only SIGUSR2 | keep the default SIGUSR2 (vLLM's own SIGUSR2 handler is refused, with a warning) |
| after a restart `/health` times out, the listen queue fills | the forked engine core shares the API server's epoll fd and listening socket; DMTCP restores a shared fd in the "leader" process, and when the child leads, its fork-time `epoll_ctl` table refills the epoll (uvloop's later watches gone), sometimes without `listen` | fdkeep records each epoll's watches from `/proc/self/fdinfo` (those whose fd in this process is the watched file) and every `SO_ACCEPTCONN` socket; after the restart it adds back the missing watches and listens again (raw syscalls). It fired in 3 of 4 restarts |

## Numbers (A40, Qwen2.5-0.5B, plain `vllm serve`, before the warmctl integration)

The image is 18 GB: API server 5.3 GB and engine core 13.9 GB, with weights and KV cache not exported.
Four restarts in a row all gave the same answer at temperature 0.

| Start | Seconds to `/health` |
|---|---|
| vLLM, empty caches | 133–155 (80 of them FlashInfer's JIT sampling build, 13 torch.compile) |
| vLLM, warm caches (warmctl's cache env) | 34 |
| `dmtcp_restart`, image on disk, page cache evicted | 14.8–18.4 (restart 12.5–14.1, CUDA 2.3–4.3) |
| `dmtcp_restart`, image in the page cache | 10.5–13.8 (restart 8.5–11.3, CUDA 2.0–2.5) |

`dmtcp_restart` reads each image with buffered single-stream reads. On RunPod's network volume (NFS 4.1
on VAST) that is about 1 GB/s, against 5.7 GB/s for parallel direct reads. The restore's prefetch reads
the `criu/` files in parallel first, and it covers DMTCP's images too.

## Limits

- Interposed IPC (cuInterpose) is not supported, and neither is TP>1: NCCL sockets, `/dev/shm` queues
  and anonymous shared memory are untested. DMTCP warns that it cannot restore anonymous shared memory.
- `io_uring` cannot be disabled by a pod's sysctl. The engine runs with `UV_USE_IO_URING=0`.
- DMTCP 5.0 must be built from source (`internal/dmtcp/fdkeep/` holds the plugin):

```bash
git clone --depth 1 https://github.com/dmtcp/dmtcp && cd dmtcp && ./configure --prefix=/usr/local && make -j && make install
gcc -shared -fPIC -O2 -I/usr/local/include -o libdmtcp_fdkeep.so internal/dmtcp/fdkeep/fdkeep.c
```
