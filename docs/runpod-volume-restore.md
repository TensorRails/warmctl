# Fresh H200 replicas from a prepared network volume

On 2026-11-01, two fresh Runpod pods returned their first external streaming
token in **52.14 s and 52.94 s** from the pod-creation API request. Both
included deliberate client-cache eviction advice for the test. An ordinary
fresh-pod run without that advice returned its first token in **32.93 s**,
with host cache still warm. The released
warmctl **v2.0.0-beta.17** and TensorRails DMTCP **v5.0.0-warmctl.1** restored
the same verified snapshot directly from NFS; no rebake or image build ran
on the receiving pods.

These are measured runs, not a provisioning or latency SLA. Numerical
equivalence is still unqualified: the canary reported differences at its
unchanged 0.001 tolerance under the profile's `warn` policy. Weight
fingerprints, streaming, tool calls and 32 concurrent short requests passed
on each replica. Each pod was deleted after its acceptance checks.

## Tested configuration

- Qwen3.8-Flash-Next-NVFP4, TP1, 32 sequences, 32,768-token context,
  48 GiB KV cache, Marlin and three speculative MTP tokens; vLLM 0.32.0,
  Torch 2.13.0+cu130, Python 3.12.3.
- One H200, NVIDIA driver 595.91.07, US-CA-2 high-performance NFS4.1
  volume, mounted at `/nv` rather than `/workspace`.
- Placement filters `minVcpuCountPerGpu: 27`, `minRamPerGpu: 240`.
  The measured allocation had 27 advertised vCPUs, a 22.95-core cgroup
  quota and 287 GB RAM. A 32-vCPU placement was unavailable.
- Cached stock image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`.
  Runpod resolved it to
  `rcr.runpod.io/runpod/pytorch@sha256:0a360022e8de4375af99430f84e8b38951acc397252163a37ceac7204d01be35`.
  Each pull log was checked. The same Docker Hub tag resolved differently;
  do not assume the provider's cache and Docker Hub are interchangeable.
- Snapshot: 134.04 GiB in 98 files, including 76.50 GiB of exported weights.
  Its largest DMTCP image is 53.59 GiB. `process.needs_model` is false:
  restore does not reload the original safetensors.

The runtime images were also published as `v2.0.0-beta.17` and
`v2.0.0-beta.17-vllm032` at `registry.warmctl.io/warmctl/runtime`. Those
images passed CPU smoke tests. These GPU measurements use the stock-image
route and do not qualify custom-image pull/start latency.

## Prepare once

1. Install the pinned warmctl release in `/opt/warmctl`, including its
   versioned DMTCP runtime. Build the Python environment at the path the
   donor will use, `/opt/warmctl-python`. Keep the base image, system
   libraries, GPU driver and CPU features compatible on receivers.
2. Save the installed warmctl tree and Python environment as PAX tar
   archives on the network volume. PAX preserves the nanosecond mtimes
   that DMTCP checks. Verify the archive digests. Extract the Python tree
   once into a versioned volume directory and compare it against the
   verified archive before admitting replicas. Do not run upgrades in
   this shared tree; deploy a new version and rebake when required.
3. Build the snapshot on local disk with `warmctl build --in-place
   --backend dmtcp --output LOCAL_SNAPSHOT ...`, then run
   `warmctl verify LOCAL_SNAPSHOT --deep`.
4. Copy the complete snapshot to the volume with checked hashes and
   per-file fsync, committing its manifest last. Retain the local copy
   and recovery manifest until `warmctl verify NETWORK_SNAPSHOT --deep`
   passes. The measured 134.04 GiB copy took 107.06 s.

Direct-to-NFS publication failed once with EIO after checkpoint creation.
Its cleanup discarded the incomplete output and donor working directory;
[issue #67](https://github.com/TensorRails/warmctl/issues/67) tracks recovery
and diagnostics. The local-first procedure above subsequently passed both
deep verifications. Stopping a Runpod pod does not preserve its ephemeral
container disk; only shut down after the network copy is verified.

## Start each replica

Use a fresh container with an init process, and reserve `/workspace`, `/tmp`
and `/dev/shm` for warmctl's in-place runtime. Preserve image startup files
outside those roots before claiming them. Attach the prepared volume at
`/nv` and select the compatible stock image and hardware.

Extract the small prepared warmctl tree back to `/opt/warmctl`. Link the
already verified Python tree from the volume at `/opt/warmctl-python`;
this preserves the donor's paths without extracting 8 GiB on every pod.
In these three runs this bootstrap took 0.40–0.58 s, versus 43.90 s for
checking and extracting the full Python archive.

Run the released CLI with:

```sh
export PATH=/opt/warmctl-python/bin:/opt/warmctl:$PATH
export WARMCTL_PREFETCH_WORKERS=16
export WARMCTL_PREFETCH_ORDER=together
export DMTCP_RESTART_HUGEPAGES=1
export DMTCP_RESTART_READERS=16
export DMTCP_RESTART_MMAP=0
warmctl up "$NETWORK_SNAPSHOT" --mode in-place --listen 0.0.0.0:8080
```

Full SHA-256 verification and Python-tree deployment belong to preparation.
Do not repeat them or deliberately evict caches in the normal serving path.
Restore still checks weight fingerprints and executes the configured canary.

These prefetch settings are deployment tuning, not changed global defaults.
On a 20.4-core allocation, 32 readers prioritizing process images restored
in 50.20 s. Raising prefetch to 64 readers and reading both lanes together
took 55.72 s: total prefetch improved, but filesystem setup slowed from
4.66 s to 11.30 s. Sixteen readers together improved one run to 45.93 s,
but another took 63.07 s with more CPU throttling. The larger CPU
allocation and repeat measurements therefore matter.

## Measurement boundaries

| Phase | Run 09 | Run 10 | Run 12, cache unchanged |
| --- | ---: | ---: | ---: |
| Test-only client cache advice | 5.57 s | 7.28 s | Omitted |
| Prepared runtime bootstrap | 0.50 s | 0.58 s | 0.40 s |
| Filesystem restore | 3.90 s | 3.99 s | 1.34 s |
| DMTCP | 21.31 s | 21.23 s | 6.62 s |
| CUDA | 6.33 s | 6.25 s | 8.16 s |
| Weight reload | 5.3 s | 5.5 s | 5.1 s |
| warmctl to serving, including remaining checks | 38.47 s | 38.69 s | 22.82 s |
| Pod-create request to first external token | **52.14 s** | **52.94 s** | **32.93 s** |

The last row uses one client monotonic clock and includes provisioning,
test setup, restore and the HTTPS request. Phase timing comes from runtime
logs; the rows are not additive because some work overlaps. Every attempt
created a new pod. These three runs reused the same physical host;
new pods do not imply cold hardware or storage-server caches.

Before these two runs, 16 workers requested `POSIX_FADV_DONTNEED` on this
experiment's 98 snapshot files and warmctl archive. It is advisory and does
not clear Python-library or server-side caches. No global cache flush was
used. Normal deployments should omit this test-only step.

Run 12 omitted cache advice and read 133.6 GiB through prefetch in 3.5 s.
That rate reflects warm cache, not NFS throughput. No initial streaming
HTTP retries were needed for this run.

Two earlier harness failures are excluded from timing claims: a cgroup-v2
diagnostic failed on a cgroup-v1 host before restore, and Runpod Cloudflare
rejected urllib's default User-Agent with 403/1010. The harness now supports
both cgroup layouts and sends `warmctl-release-test/2.0.0-beta.17`. The latter
replica passed delayed streaming/tool/32-request checks, but its first-token
timing was invalid. A later attempt, run 11, passed health but its first
streaming request returned HTTP 404; it is also excluded. Its response body
was not captured, so the cause remains unconfirmed. The harness now saves
error bodies and allows bounded retries of initial transient HTTP errors,
before any streamed token, while retaining the original measurement clock.
All allocated test pods were stopped or deleted.

## Why the donor loaded original weights for about 100 seconds

The one-time donor build loaded the original safetensors in 99.56 s. Its
background prefetch took 91.29 s: 10 of 11 files completed by about 14 s,
with the final completion at about 91 s. The installed vLLM loader used
eight workers with one sequential read task per file. The largest file was
50.03 GiB, suggesting an imbalance between tasks; per-file completion
identity was not captured, so that explanation is not proven.

Replicas do not repeat this load. The snapshot already contains exported
weights, `needs_model` is false, and warmctl prefetch divides large inputs
into 64 MiB jobs. In runs 09/10 it read 133.6 GiB in 32.7/33.2 s, overlapping
restore; the subsequent weight reload took 5.3/5.5 s.

The snapshot manifest used here has SHA-256
`88c99ee6923068c703c4d86fba6d9f266d9c578a5f6a26658dad4bff184b7cb6`.
The earlier [20.83 s local result](dmtcp.md#large-local-images-and-the-versioned-fork)
used hot local file cache and excluded provisioning and network transfer.
