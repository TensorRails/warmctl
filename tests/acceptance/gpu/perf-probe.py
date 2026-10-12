"""Benchmark a serving endpoint with `vllm bench serve` from the pinned image.
Usage: perf-probe.py BASE_URL MODEL OUT.json
Results per scenario land in OUT's directory under bench/; OUT.json holds the
summary. Env: PERF_TOKENIZER (model directory), PERF_IMAGE."""
import json
import os
import subprocess
import sys
import time

IMAGE = os.environ.get('PERF_IMAGE', 'vllm/vllm-openai:cu134-nightly@sha256:'
                       'ad9dc6397abc878d8e764daa39e16bdc7f1ecf9fb3d0283c076530b299c31ad0')
# (name, input tokens, output tokens, concurrency)
SCENARIOS = [
    ('chat', 2048, 512, 1), ('chat', 2048, 512, 16), ('chat', 2048, 512, 64),
    ('long', 32768, 512, 1), ('long', 32768, 512, 8),
    ('decode', 4096, 2048, 32),
]
KEYS = ('completed', 'failed', 'request_throughput', 'output_throughput', 'total_token_throughput',
        'median_ttft_ms', 'p95_ttft_ms', 'median_tpot_ms', 'median_itl_ms', 'p95_itl_ms', 'median_e2el_ms')


def main():
    base, model, out = sys.argv[1:4]
    bench = os.path.join(os.path.dirname(os.path.abspath(out)), 'bench')
    os.makedirs(bench, exist_ok=True)
    rows, ok = [], True
    for name, isl, osl, conc in SCENARIOS:
        label = '%s-c%d' % (name, conc)
        prompts = max(2 * conc, 8)
        cmd = ['docker', 'run', '--rm', '--network', 'host', '--entrypoint', 'vllm',
               '-e', 'HF_HUB_OFFLINE=1', '-e', 'VLLM_NO_USAGE_STATS=1',
               '-v', '%s:/tokenizer:ro' % os.environ['PERF_TOKENIZER'], '-v', '%s:/out' % bench, IMAGE,
               'bench', 'serve', '--backend', 'vllm', '--base-url', base, '--endpoint', '/v1/completions',
               '--model', model, '--tokenizer', '/tokenizer', '--dataset-name', 'random',
               '--random-input-len', str(isl), '--random-output-len', str(osl), '--random-range-ratio', '0',
               '--ignore-eos', '--num-prompts', str(prompts), '--max-concurrency', str(conc),
               '--request-rate', 'inf', '--seed', '42', '--percentile-metrics', 'ttft,tpot,itl,e2el',
               '--metric-percentiles', '50,95,99', '--save-result', '--result-dir', '/out',
               '--result-filename', label + '.json', '--disable-tqdm']
        started = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        row = {'label': label, 'exit': proc.returncode, 'seconds': round(time.time() - started, 1)}
        try:
            with open(os.path.join(bench, label + '.json')) as f:
                data = json.load(f)
            row.update({k: data.get(k) for k in KEYS})
        except (OSError, ValueError):
            row['stderr'] = proc.stderr[-1500:]
        ok = ok and proc.returncode == 0 and not row.get('failed')
        rows.append(row)
        print('%-10s out %8.1f tok/s  ttft p50 %7.1f p95 %7.1f ms  itl p50 %6.2f p95 %6.2f ms  tpot %6.2f ms' % (
            label, row.get('output_throughput') or 0, row.get('median_ttft_ms') or 0, row.get('p95_ttft_ms') or 0,
            row.get('median_itl_ms') or 0, row.get('p95_itl_ms') or 0, row.get('median_tpot_ms') or 0), flush=True)
    with open(out, 'w') as f:
        json.dump({'rows': rows, 'pass': ok}, f, indent=1)
    sys.exit(0 if ok else 1)


main()
