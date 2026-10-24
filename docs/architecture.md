# Architecture

## Pieces

- **Host CLI** (`cmd/warmctl`, `internal/{snapshot,restore,cli}`): resolves the
  profile, checks the host, runs one container, drives its agent and the
  engine, publishes or serves.
- **Distribution** (`internal/{oci,config,store}`): moves snapshots between
  the local store, registries and OCI image layouts on shared file systems.
- **Agent** (`internal/agent`): the container's main process under
  docker-init. It starts the engine (snapshot) and runs the process steps over
  a unix socket in `/state`: newline-delimited JSON, a fixed step order, no
  retries. A failed step poisons the agent; the host removes the container.
  It also bridges `/state/inference.sock` to the engine's loopback port: the
  container has no network.
- **Extension** (`python/warmctl_vllm`): mixed into vLLM's worker with
  `--worker-extension-cls`, called through `/collective_rpc`. It exports and
  reloads the sleep-mode allocator's `weights` pool and fingerprints it.

## Snapshot

| Step | Who | What |
|---|---|---|
| preflight | host | GPUs (NVML), hardware row, idle, io_uring off, GPU leases |
| start | host, agent | donor container; the agent picks the engine's ipc mode (cuInterpose or native), moves the PID allocator to 32768 and starts `vllm serve` |
| warm up | host | the profile's warmup grid through `/v1/completions`, prefix cache reset before each case |
| canary | host | one prompt twice: the miss and the hit are the baseline a restore must reproduce |
| export | extension | every `weights` allocation, fingerprinted on the GPU, written in parts of at most 1 GiB through page-locked slots, SHA-256 inline |
| sleep | host | `sleep(level=2)`; the manifest keeps the exported allocations still in the allocator table |
| prepare | extension | vLLM's `checkpoint_prepare`; whether any collective keeps peer memory mapped |
| ipc | agent | under cuInterpose: the coordinator saves every cross-process allocation's bytes and releases the sharing, addresses kept ([interpose.md](interpose.md)) |
| cuda | agent | cuda-checkpoint lock and checkpoint of every process holding CUDA state |
| freeze, dump | agent | SIGSTOP through pidfds, `criu dump` (link remaps, empty plugin directory) |
| fs | agent | `/workspace`, `/tmp`, `/dev/shm` as tar plus JSON, taken after the dump so CRIU's link_remap files are ordinary members |
| publish | host | every file into the content-addressed store, linked into the output; `manifest.json` last |

## Restore

| Step | Who | What |
|---|---|---|
| preflight | host | same GPU model, memory, driver and topology class; every CPU flag the donor had |
| launching | host | while `up` still pulls the snapshot: wait for the files the receiver starts with (adapter, cuInterpose build, driver libraries), which the pull fetches first; the engine image is pulled meanwhile |
| start | host | receiver container: the snapshot read-only at `/capture`, its Python package at `/opt/warmctl/adapter`, its cuInterpose build at `/opt/warmctl/ipc`; during a pull it starts before the page cache fills, since `docker create` waits for dirty pages to be written |
| arrive | host | while `up` still pulls the snapshot: wait until every file but the weights lane is in place and checked |
| install-fs | agent | the three trees, checked against their manifests |
| criu, thaw | agent | `criu restore --leave-stopped`, then SIGCONT through pidfds |
| cuda | agent | cuda-checkpoint restore with the donor-to-receiver device map, workers in parallel when no collective links them, then unlock |
| ipc | agent | under cuInterpose: the coordinator re-creates the shared allocations and multicast objects at their addresses and loads their bytes |
| arrived | host | while `up` still pulls the snapshot: wait for the weights lane; the reload reads the checked bytes from memory while the pull syncs them |
| wake | host, extension | vLLM's `checkpoint_restore`, `wake_up(weights)`, the pool reloaded and every allocation's fingerprint compared, `wake_up(kv_cache)` |
| canary | host | the donor's prompt twice; first must miss, second hit; compared under the profile's policy (bitwise, tolerance or warn) |
| admit | host | the gateway opens: inference routes only, 503 until now |

## What is checked, and what is not

Checked: SHA-256 of every file as it is written, pulled or (with
`--verify-weights`) reloaded; the allocation list after wake; every
allocation's fingerprint; the canary; host compatibility. Not checked: tensor
by tensor, graph by graph. Anything the export does not hold stays in the CRIU
dump, so an unknown allocation costs dump size, never correctness.

## Files

`manifest.json` (schema `warmctl.snapshot/v2`) lists every file with its size,
SHA-256 and role, the effective profile and argv, the hardware, the dumped
process tree, the CUDA owners, the weights layout with fingerprints, and the
canary.

## Distribution

A snapshot is one OCI artifact (`application/vnd.warmctl.snapshot.v2`) with
`manifest.json` as its config. Its files map onto layers in order:

| Layer | When | Checked against |
|---|---|---|
| whole file | 16 MiB up to the piece size (64 MiB) | its digest, the file's SHA-256 |
| pieces | a file over the piece size, cut in consecutive pieces | each piece's digest, then the file's SHA-256 |
| bundle | a run of files each under a quarter piece, back to back | each file's SHA-256 from `manifest.json` |

So a layer fits one request under a Workers limit and one CDN cache object,
and a 678 GiB snapshot of 6974 files is ~10,900 layers, a 2.8 MB manifest.

A model's repository holds one snapshot per platform under its platform tag
(`8xh100-80gb-sxm-580.95.05`) and lists them in `latest`, an OCI image index
whose entries carry the GPU model and count, driver, profile, vLLM and size.
`up NAME` takes the entry whose `manifest.json` passes the host check, before
any file is fetched. A namespace publishes its models in
`<namespace>/catalog:latest`.

Pulls try the cache directory (OCI image layouts on a shared file system),
the mirrors and the registry in that order, write through to the cache, and
resume: a large file assembles in `.assemble-<sha256>` next to the store,
locked by the pull, with a journal of the pieces checked. A pull fetches the
files the process restore reads first and the weights lane last, round-robin
across ranks, and links every file into the snapshot directory once its
SHA-256 is checked. `up` restores while it pulls: the restore starts once
`manifest.json` passed the host check, CRIU and CUDA run on the files that
have arrived, and the engine wakes once the weights have. A file is linked
when its bytes are checked, before the store syncs it; `manifest.json` comes
once every blob is synced.
