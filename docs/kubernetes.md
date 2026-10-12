# Kubernetes

`kubectl apply` a Deployment and a restored engine serves in seconds, on a
GPU node of the snapshot's platform, behind a Service, scaled by queue depth.
The pod is the receiver: the warmctl runtime image is the snapshot's vLLM
image with warmctl on top, and its command is the one a host runs:

```
tini -- warmctl up NAME|DIR
```

`up` sees the image's marker and restores in place, in the pod's own
container, instead of asking Docker for a receiver container.

Status: prototype. The whole path runs on a CPU host in pod-like containers
(`make e2e-test`, below); nothing has restored on a GPU node of a Kubernetes
cluster yet. [What the GPU test must show](#gpu-test).

## Design

A host restore (`warmctl up` on a Docker host) creates a receiver container
and drives the agent in it over a unix socket. In a pod, Kubernetes has
already made the container, so an in-place `up` does what a host restore
does minus Docker:

| Step | Host | In place |
|---|---|---|
| gateway | opens at admit | opens first, before the pull: `/health` 503, inference routes 503 |
| resolve | NAME, `oci://`, DIR; cache, mirrors, registry | the same, into the state directory (a node-local `hostPath`) |
| preflight | GPUs, driver, topology, CPU flags; Docker and docker-init | the same, plus [the pod's own checks](#preflight-in-the-pod); no Docker |
| receiver | `docker create`: mounts, caps, limits, docker-init as PID 1 | the pod's container, tini as PID 1; warmctl mounts tmpfs at `/tmp` and `/dev/shm` and bind-mounts the snapshot's extension, cuInterpose and (under another driver release) the donor's driver libraries read-only |
| agent | the container's main process | `warmctl agent` as warmctl's child, in a UTS namespace of its own named `warmctl`; same config, socket and steps |
| install-fs … ipc | agent steps | the same agent steps |
| wake, canary | host, through `inference.sock` | the same |
| admit | gateway opens | the gate turns ready: `/health` 200 |
| stop | `warmctl down` stops the container | SIGTERM from kubelet → tini → warmctl → agent → engine |

The agent and its step protocol are unchanged: in place replaces only the
container orchestration (`internal/run/child.go`, `internal/restore/pod.go`).
Paths the agent and the engine see are the receiver's where the dumped
processes depend on them and real paths otherwise:

| Receiver container | Pod | Why it must match |
|---|---|---|
| `/opt/warmctl/adapter` (bind) | the snapshot's `adapter/`, bind-mounted read-only over the image's copy | restored workers import the extension from it |
| `/opt/warmctl/ipc` (bind) | the snapshot's `ipc/`, bind-mounted read-only | interposed processes map `libcuinterpose*.so` from it; the coordinator reads its state |
| model at `model.container_path` | a pod volume mounted there, or `--model-dir` bind-mounted | when the dump keeps model files open |
| `/tmp`, `/dev/shm` (tmpfs) | tmpfs mounted by warmctl, the receiver's sizes | trees the snapshot carries |
| `/workspace` | the container's layer, or a volume | a tree: JIT caches, the engine's log in `/workspace/logs` |
| `/capture`, `/state`, `/work` | real paths under the state and run directories | read by the agent and by RPC arguments only |

PID 1 matters: CRIU restores the engine detached, so its root is reparented
to PID 1, which must reap. warmctl refuses to be PID 1 and refuses a PID 1
that is not a container init (tini, docker-init, dumb-init, catatonit,
pause), and refuses the node's UTS namespace, where containerd puts a
`hostNetwork` pod: there the restored engine's `127.0.0.1:8000` would be the
node's loopback. `hostNetwork` is not supported.

The engine ran under the receiver's host name, `warmctl`. The agent starts
in a UTS namespace of its own and names it so; CRIU and the restored tree
share it, while the pod keeps its own name. The agent refuses to name a UTS
namespace it shares with its parent.

A pod restarts its container in place after a crash, and a restarted
container keeps its layer and the pod's volumes, so `/workspace` and the run
directory may hold what an earlier restore left. An in-place restore writes
`.warmctl-receiver` into each before it installs and later empties only a
directory that is empty or carries that mark. Anything else, say a volume
of the operator's mounted at `/workspace`, is refused, untouched: mount an
emptyDir there or leave it to the container's layer. `/tmp` and `/dev/shm`
are fresh tmpfs every time.

The engine's own output goes to `/workspace/logs/engine.log`, as in the
receiver; after admit an in-place restore copies its new lines to stderr, so
`kubectl logs` shows the agent, the restore and the engine.

## Which mode

`up` decides before it opens anything or pulls:

1. `--mode host` or `--mode in-place` wins.
2. `/opt/warmctl/RUNTIME` exists: this is the warmctl runtime image, built
   from [deploy/kubernetes/Dockerfile](../deploy/kubernetes/Dockerfile),
   which writes it last (`warmctl=VERSION`, `base=IMAGE`). Restore in place.
3. No marker and the Docker API answers: a host restore, as before.
4. Neither: fail, "docker not found and this is not the warmctl runtime
   image: install Docker, or run the warmctl image in Kubernetes".

A missing Docker socket proves nothing (a bare host may not have Docker),
and a container proves nothing either (Docker-in-Docker, CI runners), so
neither picks the mode. Kubernetes' traces (`KUBERNETES_SERVICE_HOST`, the
service account mount, a `kubepods` cgroup) only name the pod in the log
line: `Kubernetes pod (KUBERNETES_SERVICE_HOST), warmctl runtime image
v2.0.0-beta.4 on vllm/vllm-openai@sha256:…: restoring in place`. The marker
wins over a Docker socket mounted into the image's container; `--mode host`
is the way back.

## The pod

[deploy/kubernetes/warmctl.yaml](../deploy/kubernetes/warmctl.yaml), the
whole of it but the Service:

```yaml
containers:
  - name: warmctl
    image: registry.warmctl.io/warmctl/runtime:latest
    args: [up, glm53-flash]
    securityContext: {privileged: true}
    resources:
      limits: {nvidia.com/gpu: 8}
    readinessProbe:
      httpGet: {path: /health, port: 8080}
      periodSeconds: 1
    volumeMounts:
      - {name: snapshots, mountPath: /var/lib/warmctl}
volumes:
  - name: snapshots
    hostPath: {path: /var/lib/warmctl, type: DirectoryOrCreate}
```

**Privileged** because CRIU and cuda-checkpoint need root, every
checkpoint capability, no seccomp or AppArmor profile and a writable
`/proc/sys` (the agent moves the PID allocator with `ns_last_pid`). Outside
a user namespace only `privileged: true` gives a writable `/proc/sys`
(`procMount: Unmasked` needs `hostUsers: false`). A cluster that enforces
Pod Security needs the namespace labelled
`pod-security.kubernetes.io/enforce: privileged`.

**No volumes are required.** A receiver container differs from a pod's
defaults in three ways, and warmctl, privileged anyway, fixes them in its
own container before it restores: a fresh tmpfs at `/tmp` (1 GiB, exec, for
JIT compilers) and at `/dev/shm` (the profile's `resources.shm`, 16 GiB by
default, where a pod has 64 MiB), and the host name `warmctl` the engine
ran under, in the agent's own UTS namespace. `/workspace`, the run
directory and the store live in the container's own layer.

**The one volume**, `/var/lib/warmctl` on the node, is for speed: snapshots
pulled there stay for the next pod on the node, and
[node-prep.yaml](../deploy/kubernetes/node-prep.yaml) can pull them ahead.
Without it every new pod downloads the snapshot into its own layer. Keep it
on local disk: a snapshot read from a virtiofs share restored in 63 s
instead of 14 s.

Optional, each a few lines:

- a shared file system as the cache of snapshots (`WARMCTL_CACHE` and a
  `ReadWriteMany` volume): the next node pulls at file system speed;
- `nodeSelector` on the GPU Feature Discovery labels
  (`nvidia.com/gpu.product`, `nvidia.com/gpu.count`, the driver branch) in a
  cluster with nodes of other platforms; on a node that does not match, the
  preflight refuses;
- `readOnlyRootFilesystem: true`, as the receiver has it: then
  `/workspace`, `/run/warmctl` and `/var/lib/warmctl` need volumes
  (emptyDirs); warmctl still mounts `/tmp` and `/dev/shm` itself;
- a memory limit of the profile's `resources.memory`; the tmpfs count
  against it.

### GPUs

The NVIDIA device plugin allocates `nvidia.com/gpu: N`; the NVIDIA
container toolkit puts the driver libraries and `nvidia-smi` in the
container. A privileged container sees every GPU of the node, so
an in-place restore takes the GPUs the device plugin named in
`NVIDIA_VISIBLE_DEVICES` (UUIDs or indices, or the volume-mounts strategy's
directory) and otherwise, as on a host, on the first free GPUs of the
snapshot's model. The snapshot's platform pins the node: same GPU model and
count, a driver of the same branch (580.x), NVLink topology, the donor's
CPU flags. Under another release of the branch the restore bind-mounts the
donor's driver libraries, which the snapshot carries, where the donor's
container had them, as a host restore does.

### Readiness and autoscaling

The gateway listens on `:8080` from the start (port 8000 is the restored
engine's own, on the pod's loopback). `GET /health` is 503 until the canary
passed and 200 after; a `readinessProbe` on it keeps the Service from
routing to a pod that is still restoring. Only `/v1/completions`,
`/v1/chat/completions`, `/v1/models` and `/metrics` pass.

[deploy/kubernetes/keda.yaml](../deploy/kubernetes/keda.yaml) scales the
Deployment on `vllm:num_requests_waiting` scraped from the pods' `/metrics`
(Prometheus and KEDA installed). A new replica costs a restore, not a cold
start; GPU node provisioning, if the cluster has to add a node, does not
shrink.

## Node preparation

[deploy/kubernetes/node-prep.yaml](../deploy/kubernetes/node-prep.yaml), a
DaemonSet on the GPU nodes:

- `kernel.io_uring_disabled=2`: CRIU cannot dump io_uring. A restore does
  not dump, but the snapshot was taken with io_uring off and the restored
  engine should find the kernel as it was; nodes that bake need it anyway.
  The sysctl is not namespaced: a privileged container sets it for the node.
- The pod image, pulled by running it: ~20 GB of vLLM image is minutes, not
  seconds, so it must be on the node before the Deployment lands there.
- The snapshot, pulled by platform tag into the node's `/var/lib/warmctl`
  and read once, so a restore hits the page cache (a cold page cache cost
  GLM 186 s against 37 s warm). `up` then finds it pulled before.

## The image

The published runtime image is `registry.warmctl.io/warmctl/runtime:latest`,
pulled anonymously. `:latest` follows the newest release; each release is also
tagged by version (`:v2.0.0-beta.5`), which a cluster that must not change
under it pins. To run a copy in your own registry, or a runtime image on
another vLLM base, build it:

```
make pod-image POD_IMAGE=<registry>/warmctl/runtime:<tag>   # bin/cuinterpose, then deploy/kubernetes/Dockerfile
docker push <registry>/warmctl/runtime:<tag>
```

The image's command is `up` and its last layer the marker
`/opt/warmctl/RUNTIME`; a pod's `args` are the host's command line
(`[up, glm53-flash]`).

`BASE` (default: the image pinned in `profiles/_defaults.yaml`) must be the
snapshot's `launch.image`: every library the restored processes map comes
from it. The image's marker records its base and an in-place restore
refuses a snapshot taken in another image. One runtime image serves
every snapshot of that vLLM image.

## Preflight in the pod

Before anything is mounted or started, an in-place restore checks:

- PID 1 is a container init and warmctl is neither PID 1 nor in the
  snapshot's PID range (≥ 32768);
- the checkpoint capabilities are effective and no seccomp filter applies;
- `/proc/sys/kernel/ns_last_pid` is writable;
- the mount points exist (the image was built from the Dockerfile here);
- `/workspace` is writable and large enough for its archive;
- the model is mounted at `model.container_path` when the dump needs it;
- the image's base is the snapshot's image;
- the pod is not in the node's UTS namespace (`hostNetwork`);
- `/workspace` and the run directory are empty or a restore's (the mark);
- and, as on a host, the GPUs, driver branch, topology and CPU flags.

It warns when the memory limit is below the profile's, or, under the
donor's own driver release, when a driver library the dump maps is not
where the donor's container had it. Then it mounts the tmpfs and starts the
agent in its own UTS namespace.

## Open risks

- **Docker-built snapshots restored under containerd.** Every snapshot so
  far was baked in a Docker container with the NVIDIA runtime's legacy
  hook, which puts the driver libraries in the image's library directory
  (`/usr/lib/x86_64-linux-gnu/libcuda.so.<driver>` for vLLM's Ubuntu
  image). Under containerd the toolkit may run as a CDI spec instead, which
  mounts them at the host's paths: the same on an Ubuntu node, different on
  a RHEL-family node or a driver container with another layout. Under the
  donor's own driver release a missing path fails CRIU's restore of the
  mapping (the preflight names it); under another release of the branch the
  donor's copies are mounted at the donor's paths anyway. The same goes for anything else
  the runtime injects (firmware, `nvidia-smi`, `/dev/nvidia-caps`), and for
  kubelet's own mounts (`/etc/hosts`, `/etc/hostname`, `/etc/resolv.conf`,
  `/dev/termination-log`, the service account): Docker mounted the same
  files, the dumped processes hold none of them open, but nothing has
  checked that on a GPU yet. If paths differ, bake in the cluster instead
  (an in-place `warmctl build`, not written) or align the CDI spec.
- **Privileged pods.** The namespace must allow privileged pods, and a
  privileged container sees every device of the node, so a node is in
  practice one pod (or pods that split its GPUs through
  `NVIDIA_VISIBLE_DEVICES`, untested). Without privileges the path is user
  namespaces (`hostUsers: false` with `procMount: Unmasked`), which CRIU,
  cuda-checkpoint and the NVIDIA devices have not been tried under.
- **Single node.** A snapshot is one host's engine: tensor parallel within
  one node, no pipeline or data parallel across nodes. A pod is one node of
  the snapshot's platform: GPU model and count, driver branch (an upgrade
  by the GPU Operator to another branch invalidates every snapshot of the
  node pool), topology class and CPU flags.
- **hostNetwork.** Unsupported and refused: the pod would share the node's
  network and UTS namespaces, the restored engine's `127.0.0.1:8000` would
  be the node's loopback.
- **The pod's network namespace.** The receiver container had only
  loopback; a pod has the pod network too. The restored engine listens on
  `127.0.0.1:8000` and its ranks talk over loopback TCP, which CRIU restores
  (`--tcp-established`, with iptables in the image). Sidecars that bind
  loopback ports (service meshes) can collide with the restored sockets;
  run the pod without them.
- **cgroups.** CRIU restores the tree into the container's cgroup (cgroup
  v2 with a private cgroup namespace, containerd's default). A cgroup v1
  node is untested.
- **Engine log growth.** The engine keeps appending to
  `/workspace/logs/engine.log` in the container.

## CPU test

`make e2e-test` builds the runtime image on the stand-in vLLM image
(`--build-arg BASE=warmctl-e2e:local`) and, after the host restores, runs it
with the host's command line, `up NAME|DIR` and no mode flag, as
warmctl.yaml runs a pod: `docker run --privileged` and nothing else of
Docker's (a 64 MiB `/dev/shm`, a host name of its own), its own bridge
network with the gateway published, no Docker socket, tini from the
entrypoint, the model read-only at `/model`. Every restore must pick in
place from the marker, name the Kubernetes pod when
`KUBERNETES_SERVICE_HOST` is set, and run the engine under the host name
`warmctl` while the container keeps its own. The same container without
`--privileged` must be refused by the preflight, naming `privileged: true`,
and one with a file of its own in a volume at `/workspace` refused with the
file untouched. Then it restores three ways and
checks each: the readiness probe sees 503 then 200, a completion goes
through, the engine's request log reaches `docker logs`, a management route
is refused, `docker stop` (SIGTERM through tini) exits 0:

1. a snapshot directory on a read-only volume, then `docker restart` in
   place: the container's layer keeps what the first restore left in
   `/workspace`, the second empties it and restores again;
2. `fake-tp2-short` by name from a shared cache (`WARMCTL_CACHE`), the
   registry unreachable;
3. `fake-tp2-short` by name from the registry over the bridge network, in a
   hardened pod (read-only root; `/workspace`, `/run/warmctl` and the store
   on volumes; memory and PID limits), restarted with its volumes.

What a CPU cannot stand in for: CUDA (cuda-checkpoint, the NVIDIA runtime's
mounts), cuInterpose, the vLLM image itself.

## GPU test

On a managed Kubernetes cluster, one 8×H100 node group with the GPU Operator:

1. Check the node's driver is of the snapshot's branch (`nvidia-smi` on the
   node, the platform tag; `warmctl catalog --all`), and which mode the
   toolkit runs in (legacy hook or CDI; `kubectl get clusterpolicy -o
   yaml`, `cdi.enabled`).
2. Apply `node-prep.yaml` (it pulls `runtime:latest`); wait for the pull.
3. Apply `warmctl.yaml`; read `kubectl logs`: preflight warnings (driver
   library path, memory), CRIU's restore log on failure.
4. Pass: `/health` 200, the canary bitwise as on the host, the restore time
   against the host's 37 s with a warm page cache, and a completion
   through the Service.
5. Scale to two replicas on two nodes, then delete a pod and time its
   replacement; with KEDA, a load test that crosses the queue threshold.

If CRIU fails on a library path, compare the donor's mappings
(`crit show criu/mm-*.img` in the snapshot) with the pod's mounts.
