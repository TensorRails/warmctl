# Demo: GLM-5.3-Flash on Kubernetes in about a minute

Three replicas of GLM-5.3-Flash (8×H100 each) go from `kubectl apply` to answering in about a
minute, restored from a fast shared file system instead of a 650-second vLLM cold start. Then
a load test through the Service, and a killed pod that comes back on its own while traffic
keeps flowing.

| Step | On screen | Takes |
|---|---|---|
| 1 | Mirror the model onto the shared file system | seconds (done ahead, re-run live) |
| 2 | Runtime image on every GPU node | done ahead |
| 3 | `kubectl apply`, watch three replicas restore and answer | ~1 min |
| 4 | Load test through the Service | ~2 min |
| 5 | Kill a pod, traffic keeps flowing, the pod comes back | ~1 min |

Files: [glm53.yaml](glm53.yaml) (namespace, Deployment, Service), [node-prep.yaml](node-prep.yaml)
(image on the nodes), [watch.py](watch.py) (live view of the replicas), [traffic.sh](traffic.sh)
(steady requests through the Service), [bench.yaml](bench.yaml) (load test Job).

## What you need

- **Cluster:** three or more nodes with 8×H100 SXM, all on the same NVIDIA driver branch.
  - The demo uses the 580.x snapshot; use the 610.x one for 610 nodes.
  - The NVIDIA GPU Operator, or the device plugin plus GPU Feature Discovery: the manifests select `nvidia.com/gpu.product=NVIDIA-H100-80GB-HBM3`.
  - Privileged pods allowed in the `warmctl-demo` namespace.
- **Shared file system:** a fast one (NFS over RDMA, Weka, VAST, a cloud shared file system), mounted at `/mnt/share` on every GPU node and on one VM. If it is mounted elsewhere, change `/mnt/share` in `glm53.yaml`.
- **The VM:** [warmctl installed](https://warmctl.io/docs/install/) and access to `registry.warmctl.io`. No GPU needed.
- **Your laptop:** `kubectl`, `python3`. Lens or k9s are optional, for a second view.
- **Load test:** the cluster reaches the Hugging Face Hub, for the tokenizer.

> **Rehearse first.** Restoring inside a Kubernetes pod is the newest path in warmctl (see
> [docs/kubernetes.md](../docs/kubernetes.md)). Run the whole demo once on the same cluster before
> showing it.

## Before the demo

**Mirror the snapshot onto the share** (on the VM, ~10 minutes for 404 GiB from the registry):

```bash
warmctl mirror --all-platforms glm53-flash:8xh100-80gb-sxm-580.173.02 /mnt/share/warmctl
```

- `--all-platforms` lets a host without GPUs copy a platform it cannot run itself.
- The platform tag picks the 8×H100 snapshot for driver 580.x. `warmctl catalog --all` lists the others, for example `…-610.57.04`.
- The copy is an OCI image layout under `/mnt/share/warmctl/glm53-flash`, with an index, so `warmctl up glm53-flash` finds it by name.

**Put the runtime image on the nodes** (~20 GB of vLLM and warmctl per node, a few minutes):

```bash
kubectl apply -f demo/node-prep.yaml
kubectl -n warmctl-demo rollout status ds/warmctl-node-prep
```

**Start clean.** Each node keeps the snapshots it restored in `/var/lib/warmctl`. To show the copy from the share again in a later rehearsal, delete them on each node:
- `kubectl debug node/<node> -it --image=busybox -- rm -rf /host/var/lib/warmctl/pulled`
- then drop the page cache.

Without that, the next run restores from the node's own copy, which is faster but skips step 1's point.

## 1 · Mirror: put the model next to the GPUs

On the VM:

```bash
ssh demo-vm
warmctl mirror --all-platforms glm53-flash:8xh100-80gb-sxm-580.173.02 /mnt/share/warmctl
du -sh /mnt/share/warmctl/glm53-flash
```

The copy is already there, so `mirror` checks every blob and copies nothing. To say:

> GLM-5.3-Flash as a warm engine is a 404 GiB snapshot: the processes, the GPU memory and the
> CUDA state of all eight ranks, compiled kernels, CUDA graphs, and the weights. Before a GPU
> needs it, we copy it once from the registry onto the cluster's fast shared file system, so a
> node reads it at storage speed, not internet speed. Run it again and it copies only what
> changed.

## 2 · The runtime image is on the nodes

```bash
kubectl -n warmctl-demo get pods -o wide -l app=warmctl-node-prep
```

> Each GPU node already has the runtime image: vLLM and warmctl. The model is not on the nodes yet.

## 3 · Deploy and watch it come up

In one terminal, the live view:

```bash
demo/watch.py
```

In another:

```bash
kubectl apply -f demo/glm53.yaml
```

What `watch.py` shows, one row per pod:
- which node the pod is on;
- `Pending`, then `Restoring` with the restore's latest log line;
- `Ready` with the seconds from pod creation;
- the pod's first answer and its time to first token, from inside the pod.

At the bottom, how long each replica took. Lens or k9s show the same pods going ready.

> Each pod copies the snapshot from the share to the node's local disk and restores the engine
> from it: processes, GPU memory, the CUDA graphs vLLM captured and the kernels it compiled.
> No weights loading, no torch.compile, no graph capture. A vLLM cold start of the same engine
> takes 650 seconds. Here three replicas on three nodes are serving in about a minute, and each
> answers on its own.

Check the Service from the client pod:

```bash
kubectl -n warmctl-demo run curl --rm -it --image=curlimages/curl:8.10.1 --restart=Never -- \
  curl -s http://glm53-flash/v1/models
```

## 4 · Load test through the Service

```bash
kubectl apply -f demo/bench.yaml
kubectl -n warmctl-demo logs -f job/glm53-bench
```

The Job runs 192 concurrent requests with 1K-token prompts and 512-token answers, 64 per replica, through the Service. It prints:
- output tokens per second;
- time to first token;
- time per output token.

> This is the same engine as a cold-started one: same kernels, same CUDA graphs, same throughput.

## 5 · Kill a pod

Terminal 1 keeps `watch.py`. In terminal 2, steady traffic:

```bash
demo/traffic.sh
```

In terminal 3:

```bash
kubectl -n warmctl-demo delete pod "$(kubectl -n warmctl-demo get pods -l app=glm53-flash -o name | head -1 | cut -d/ -f2)"
```

What you see:
- The pod goes `Terminating`. The Deployment creates a new one on the same node, the only free 8-GPU node.
- The new pod restores from the snapshot the node kept locally, with no copy from the share this time, and answers in well under a minute.
- `traffic.sh` keeps getting answers from the other two replicas. One or two requests that were running on the killed pod can fail.

> A crashed or evicted replica is a restore, not a cold start. The same goes for scale-out: a new
> replica on a node that has the snapshot is back in seconds.

## Clean up

```bash
kubectl -n warmctl-demo delete job glm53-bench pod warmctl-client --ignore-not-found
kubectl delete -f demo/glm53.yaml
```

`node-prep.yaml` and the share can stay for the next run.

## If something goes wrong

| Symptom | Look at |
|---|---|
| Pod `Pending` | `kubectl -n warmctl-demo describe pod`: no free node with 8 GPUs, or the `nvidia.com/gpu.product` label differs |
| Pod restarts during the restore | `kubectl -n warmctl-demo logs <pod> --previous`. The preflight names what does not match: driver branch, GPUs, privileges |
| Restore pulls from `registry.warmctl.io` instead of the share | the share is not at `/mnt/share` on that node, or the mirrored platform is not the node's driver branch |
| `watch.py` shows `error:` for an answer | the gateway is up but the request failed. Run `kubectl exec` into the pod and `curl localhost:8080/health` |
| Load test fails at start | no Hugging Face access for the tokenizer. Set `HF_TOKEN`, or point `--tokenizer` at a local copy |
