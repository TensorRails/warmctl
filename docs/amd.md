# AMD Instinct GPUs (preview)

warmctl snapshots and restores a warmed vLLM engine on AMD Instinct GPUs,
in an unprivileged container, without CRIU's privileges. Nothing is asked of
the user: on an AMD host warmctl finds the GPUs itself, and a snapshot of
AMD GPUs says so in its manifest.

```bash
# In the ROCm runtime image (deploy/rocm/Dockerfile), on a pod with one MI350X:
warmctl build --profile qwen3-8b-mi350x-tp1 --model-dir /models/qwen3-8b --output /snap/q8
# In another pod of the same image, GPU model and driver:
warmctl up /snap/q8
```

## How it works

There is no `cuda-checkpoint` for AMD. The kernel's own checkpoint interface
(`AMDKFD_IOC_CRIU_OP`, which CRIU's `amdgpu` plugin uses) needs
`CAP_CHECKPOINT_RESTORE` or `CAP_SYS_ADMIN` in the initial user namespace,
which no ordinary container has. Instead, the engine's processes preload
[libkfdreplay](https://github.com/TensorRails/libkfdreplay), which records
every GPU object the ROCm runtime creates (buffers, their mappings, events,
queues) and can replay them in a new process at the same addresses.

| Step | NVIDIA | AMD |
|---|---|---|
| start | the engine under `dmtcp_launch` (or CRIU's PID floor) | the same under DMTCP, with libkfdreplay in the engine's `LD_PRELOAD` |
| warm up, canary, export, sleep | through the engine | the same |
| GPU checkpoint | the agent, over the processes' PIDs (`cuda-checkpoint`) | each worker itself, asked by warmctl through `/collective_rpc` (`warmctl_gpu_checkpoint` → `kfdr_checkpoint(1)`): GPU memory to host memory, the runtime's threads parked, the devices released |
| check | the CUDA states | the agent checks that no process of the tree still holds `/dev/kfd` or a render node, that every process running ROCm has libkfdreplay, and that the snapshot carries every file the processes map |
| dump, restart | DMTCP (or CRIU) | DMTCP |
| GPU restore | the agent, with a device map by UUID | each worker itself (`warmctl_gpu_restore` → `kfdr_restore()`), before the engine wakes; on another GPU of the same model libkfdreplay translates its `gpu_id` |
| wake, reload, canary, admit | through the engine | the same |

The snapshot's files, its distribution (OCI layers, index, catalog), the
weights export and reload with fingerprints, the canary and the gateway are
the NVIDIA path's.

## Choosing the vendor

| | Decided by |
|---|---|
| `build` | the host: AMD where no NVIDIA device exists and `/dev/kfd` lists GPUs warmctl may read; NVIDIA everywhere else, as before. On an AMD host the bake runs in place under DMTCP (implied in the runtime image), and `profiles/_defaults-rocm.yaml` is laid over the defaults: the ROCm environment libkfdreplay needs. Any profile works if it runs on one GPU. |
| manifest | `hardware.vendor: amd` and the schema `warmctl.snapshot/v2-rocm`. Snapshots without the field are NVIDIA's. Releases before AMD support refuse the schema, so they skip AMD entries of a repository's index instead of picking one. |
| `up` | the snapshot: an AMD snapshot restores only on AMD GPUs, in place, and an NVIDIA one only on NVIDIA GPUs. `up NAME` takes the index entry this host can restore. |

## What a restore needs

A DMTCP restart maps every unmodified file area again from its file, after
checking the file's size and modification time. So the receiver must have,
at the same paths:

- **the same runtime image** (its digest): ROCm, PyTorch, vLLM, DMTCP and
  libkfdreplay (`/opt/warmctl/kfdreplay/<release>/`, recorded in
  `process.gpu_lib` with its size, mtime and SHA-256);
- **the same model files** at the same path (the same volume): the ones the
  processes mapped are recorded in `process.mapped` and checked before the
  restart;
- everything else the engine wrote: compile and JIT caches, `/tmp` and
  `/dev/shm` travel in the snapshot's trees, with their modification times to
  the nanosecond. `_defaults-rocm.yaml` points ROCm's caches (MIOpen) under
  `/workspace`; a bake refuses an engine that maps a file none of these
  covers, and names it.

And: the same GPU model (ISA) and the same amdgpu driver release. Another GPU
of the host is fine when the container sees only that one: libkfdreplay
translates the GPU's `gpu_id` and reopens its render node.

## Verified

| | |
|---|---|
| GPU | MI350X (gfx950), one per engine |
| Driver | amdgpu DKMS 7.1.3 |
| ROCm | 10.1 (TheRock), PyTorch 2.13, vLLM 0.29 (`rocm/vllm` image, `_defaults-rocm.yaml`) |
| libkfdreplay | v0.1.0 (`internal/kfdreplay`) |
| DMTCP | `v5.0.0-warmctl.2` |

A bake on another ISA is refused; on another driver release it warns.

## Limits

- **One GPU per engine.** Tensor parallelism needs cross-process GPU memory
  (RCCL's), which libkfdreplay does not replay yet; a profile with more ranks
  is refused.
- **Snapshot a freshly warmed engine.** A restore brings each GPU queue back
  by replaying as many packets as it had taken (~1 µs each); a long-running
  server's queues would take minutes. A bake always starts its engine, and
  warns when a queue took more than 10⁷ packets. A restored engine is not
  snapshotted again.
- **The same host's GPUs.** Restores on another GPU of the same host are
  verified; on another host of the same model, driver and image they are
  expected to work and not yet verified.
- **SDMA queues are not replayed:** `HSA_ENABLE_SDMA=0` (ROCr copies with
  blit kernels instead).
- **Large images:** the DMTCP image is the process tree's whole memory
  (8.4 GB for Qwen3-0.6B, 23 GB for Qwen3-8B on vLLM 0.29, which keeps the
  CUDA-graph pool in VRAM while asleep).
- libkfdreplay mirrors how the ROCm runtime uses the driver. It refuses
  rather than produce a broken image when it meets a call it does not
  model; a new ROCm release may need a new libkfdreplay.

## Testing without a GPU

`make amd-e2e-test` bakes and restores in unprivileged containers of a
stand-in runtime image with the real DMTCP and libkfdreplay releases, on a
KFD topology of MI350X GPUs and a stand-in vLLM whose "GPU memory" is host
memory: the vendor's detection, the profile layer, the libkfdreplay calls,
the agent's checks, the manifest, a restore on the same and on another GPU,
and the refusals.
