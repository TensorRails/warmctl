# GPU acceptance

A change the CPU tests cannot stand in for is proven on GPUs with test lanes
(`tests/acceptance/gpu/lane.sh`). A lane builds a snapshot and restores it
several times, each restore answering a completion and the quality probe.
Lanes on disjoint GPUs run in parallel; GPU leases make a lane started on busy
GPUs fail at preflight instead of colliding.

## Before running

- `make test pytest lint criu-test e2e-test` green; `make release` (the
  version comes from the git tag).
- Models downloaded ahead: Hugging Face serves one file per connection at
  25–50 MB/s, so a model with a few large shards is the slow one.
- Snapshots, state and models on a local disk. Read from a virtiofs share, a
  gpt-oss-120b restore took 63 s instead of 14 s.
- The vLLM image pulled; `kernel.io_uring_disabled=2`.

## Running

```
R=/var/lib/warmctl-run   # binary as $R/warmctl, lane.sh, timings.py
STATE=/mnt/local/state OUTDIR=/mnt/local/snap \
  $R/lane.sh q8 qwen3-8b-tp1 /mnt/local/models/qwen3-8b 0 8001 2
python3 $R/timings.py $R/diag $R/timings.json
```

Start each lane detached (`sudo -b nohup …`). To chain lanes, wait on the
process of the lane before (`pgrep -f "lane[.]sh ID "`; the bracket keeps the
pattern out of the waiting command's own command line). Never edit a script
while bash runs it: write a new file.

`tests/acceptance/gpu/perf-probe.py` benchmarks a served engine with `vllm
bench serve`, for an A/B of a restored engine against stock vLLM with the same
flags.

## Bars

A model passes when the build exits 0, every restore prints `weights
reloaded and fingerprints match` and `serving started`, the completion
returns 200 and the probe exits 0. Reference restore times, warm page cache:

| Profile | Restore |
|---|---|
| qwen3-8b-tp1 | 8.6–9.5 s |
| gpt-oss-120b-tp2 | 13.9 s |
| qwen3-0.6b-fastcomm-tp8 (NVLS) | 19.9–21.4 s |
| glm53-flash-prod-tp8 | 37.3–37.5 s, A/B with stock passes |
| deepseek-v41-flash-prod-tp8 | 74.6–86.2 s |
