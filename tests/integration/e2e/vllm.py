#!/usr/bin/env python3
"""A stand-in for `vllm serve` on a CPU host, for warmctl's end-to-end test.

It keeps what warmctl touches in vLLM: an OpenAI-style HTTP API on loopback,
the sleep-mode lifecycle, /collective_rpc to one worker process per rank and,
in every worker, a weights pool at fixed addresses. Each worker runs
warmctl's own extension (warmctl_vllm.worker.Extension) over a pool held in
host memory, so export, reload and fingerprints are the real code paths.
Completions are a function of every rank's pool bytes: a restore that
brought back wrong bytes answers differently.
"""
import ctypes
import hashlib
import http.server
import json
import multiprocessing
import os
import random
import sys
import threading
from urllib.parse import parse_qs, urlparse

from warmctl_vllm import pool
from warmctl_vllm.worker import Extension

BASE = 0x7F0000000000
# The first allocation is one part long and the same on every rank, like a
# replicated norm: its part has the same digest on all ranks (v1 P148).
SIZES = (300000, 192 << 10, 512 << 10)
DEFERRABLE = ('weights', 'kv_cache')


class Data:
    """vLLM's AllocationData, as the extension reads it."""

    def __init__(self, size, tag):
        self.handle = (None, size)
        self.tag = tag
        self.is_asleep = False


class Device:
    """Device memory of one rank: one bytearray per mapped allocation."""

    def __init__(self, rank, seed):
        self.table, self.memory = {}, {}
        # Every rank is its own process with the same addresses (v1 P148).
        address = BASE
        for index, size in enumerate(SIZES):
            owner = 'all' if index == 0 else rank
            self.table[address] = Data(size, 'weights')
            self.memory[address] = bytearray(random.Random('%s/%s/%d' % (seed, owner, index)).randbytes(size))
            address += size + (1 << 20)
        self.table[address] = Data(64 << 10, 'kv_cache')
        self.memory[address] = bytearray(64 << 10)

    def _locate(self, ptr, n):
        for base, data in self.table.items():
            if base <= ptr and ptr + n <= base + data.handle[1]:
                buffer = self.memory.get(base)
                if buffer is None:
                    raise ValueError('allocation %#x is asleep' % base)
                return ctypes.addressof(ctypes.c_char.from_buffer(buffer, ptr - base))
        raise ValueError('no allocation holds %#x+%d' % (ptr, n))

    def bind(self):
        pass

    def register(self, ptr, n):
        return True

    def unregister(self, ptr):
        pass

    def to_host(self, host_ptr, dev_ptr, n):
        ctypes.memmove(host_ptr, self._locate(dev_ptr, n), n)

    def to_device(self, dev_ptr, host_ptr, n):
        ctypes.memmove(self._locate(dev_ptr, n), host_ptr, n)

    def sums(self, regions):
        return [pool.reference_sum(self.memory[ptr][:n], position) for ptr, n, position in regions]

    def sleep(self):
        for base, data in self.table.items():
            if data.tag in DEFERRABLE:
                data.is_asleep, self.memory[base] = True, None

    def wake(self, tags):
        for base, data in self.table.items():
            if data.is_asleep and (not tags or data.tag in tags or data.tag not in DEFERRABLE):
                data.is_asleep, self.memory[base] = False, bytearray(data.handle[1])

    def digest(self):
        h = hashlib.sha256()
        for base in sorted(self.table):
            if self.table[base].tag == 'weights' and self.memory[base] is not None:
                h.update(self.memory[base])
        return h.hexdigest()


class Worker(Extension):
    def __init__(self, rank, seed):
        self.rank, self.fake = rank, Device(rank, seed)

    def _warmctl_rank(self):
        return self.rank

    def _warmctl_table(self):
        return self.fake.table

    def _warmctl_device(self):
        return self.fake

    def _warmctl_collectives(self):
        return []

    def warmctl_describe(self):
        return {'rank': self.rank, 'pid': os.getpid(), 'device': self.rank, 'name': 'Fake GPU', 'arch': 'sm_00',
                'pci_bus_id': '0000:%02x:00.0' % (self.rank + 1), 'uuid': '', 'vllm': 'fake', 'torch': 'none',
                'adapter_sha256': os.environ.get('WARMCTL_ADAPTER_SHA256', '')}

    def checkpoint_prepare(self):
        return None

    def checkpoint_restore(self):
        return None


def serve_worker(rank, seed, conn):
    worker = Worker(rank, seed)
    while True:
        method, kwargs = conn.recv()
        try:
            if method == 'sleep':
                result = worker.fake.sleep()
            elif method == 'wake':
                result = worker.fake.wake(kwargs['tags'])
            elif method == 'digest':
                result = worker.fake.digest()
            elif method == 'asleep':
                result = any(d.is_asleep for d in worker.fake.table.values())
            else:
                result = getattr(worker, method)(**kwargs)
            conn.send(('ok', result))
        except Exception as error:  # noqa: BLE001 - carried back to the API server like vLLM does
            conn.send(('error', '%s: %s' % (type(error).__name__, error)))


class Engine:
    def __init__(self, ranks, names, seed, tp):
        self.lock, self.names, self.seen, self.tp = threading.Lock(), names, set(), tp
        self.conns = []
        for rank in range(ranks):
            parent, child = multiprocessing.Pipe()
            multiprocessing.Process(target=serve_worker, args=(rank, seed, child), daemon=True).start()
            self.conns.append(parent)

    def rpc(self, method, kwargs=None):
        with self.lock:
            for conn in self.conns:
                conn.send((method, kwargs or {}))
            replies = [conn.recv() for conn in self.conns]
        errors = [value for status, value in replies if status == 'error']
        if errors:
            raise RuntimeError('; '.join(errors))
        return [value for _, value in replies]

    def complete(self, body, replica):
        prompts = body['prompt'] if isinstance(body['prompt'][0], list) else [body['prompt']]
        digest = ''.join(self.rpc('digest')).encode()
        choices, cached = [], 0
        for index, prompt in enumerate(prompts):
            key = (replica, tuple(prompt))
            if key in self.seen:
                cached += len(prompt) // 16 * 16
            self.seen.add(key)
            seed = hashlib.sha256(digest + json.dumps(prompt).encode()).digest()
            tokens, logprobs = [], []
            for j in range(body['max_tokens']):
                h = hashlib.sha256(seed + j.to_bytes(4, 'little')).digest()
                tokens.append('token_id:%d' % int.from_bytes(h[:2], 'little'))
                logprobs.append(-h[2] / 255.0)
            choices.append({'index': index, 'text': '', 'logprobs': {'tokens': tokens, 'token_logprobs': logprobs}})
        return {'choices': choices, 'usage': {'prompt_tokens': sum(len(p) for p in prompts),
                                              'prompt_tokens_details': {'cached_tokens': cached}}}


def handler(engine):
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            sys.stdout.write('http %s\n' % (fmt % args))
            sys.stdout.flush()

        def reply(self, code, value):
            raw = json.dumps(value).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == '/health':
                return self.reply(200, {})
            if path == '/is_sleeping':
                return self.reply(200, {'is_sleeping': any(engine.rpc('asleep'))})
            if path == '/v1/models':
                return self.reply(200, {'data': [{'id': name} for name in engine.names]})
            return self.reply(404, {'error': path})

        def do_POST(self):
            url = urlparse(self.path)
            length = int(self.headers.get('Content-Length') or 0)
            body = json.loads(self.rfile.read(length) or b'{}')
            try:
                if url.path == '/sleep':
                    engine.rpc('sleep')
                    engine.seen.clear()
                    return self.reply(200, {})
                if url.path == '/wake_up':
                    engine.rpc('wake', {'tags': parse_qs(url.query).get('tags', [])})
                    return self.reply(200, {})
                if url.path == '/reset_prefix_cache':
                    engine.seen.clear()
                    return self.reply(200, {})
                if url.path == '/collective_rpc':
                    # Every replica runs it; vLLM's DP frontend answers with the first one's workers.
                    results = engine.rpc(body['method'], body.get('kwargs'))
                    return self.reply(200, {'results': results[:engine.tp]})
                if url.path == '/v1/completions':
                    if any(engine.rpc('asleep')):
                        return self.reply(400, {'error': 'engine is sleeping'})
                    return self.reply(200, engine.complete(body, self.headers.get('X-data-parallel-rank', '0')))
            except Exception as error:  # noqa: BLE001 - any worker failure is an HTTP 500, as in vLLM
                return self.reply(500, {'error': str(error)})
            return self.reply(404, {'error': url.path})

    return Handler


def option(argv, name, default=None, many=False):
    if name not in argv:
        return default
    at = argv.index(name) + 1
    values = []
    while at < len(argv) and not argv[at].startswith('--'):
        values.append(argv[at])
        at += 1
    return values if many else values[0]


HELD = []


def main(argv):
    if argv[:1] != ['serve']:
        sys.exit('usage: vllm serve MODEL [flags]')
    if os.environ.get('FAKE_VLLM_FAIL') == '1':
        sys.exit('fake vllm: failing on purpose')
    model = argv[1]
    with open(os.path.join(model, 'config.json')) as f:
        seed = json.load(f)['seed']
    if os.environ.get('FAKE_VLLM_HOLD_MODEL') == '1':
        # Keep a model file open, as a loader that mmaps weights would.
        HELD.append(open(os.path.join(model, 'config.json')))
    tp = int(option(argv, '--tensor-parallel-size', '1'))
    ranks = tp * int(option(argv, '--data-parallel-size', '1'))
    names = option(argv, '--served-model-name', [model], many=True)
    engine = Engine(ranks, names, seed, tp)
    server = http.server.ThreadingHTTPServer((option(argv, '--host', '127.0.0.1'), int(option(argv, '--port', '8000'))),
                                            handler(engine))
    print('fake vllm: %d ranks serving %s' % (ranks, names), flush=True)
    server.serve_forever()


if __name__ == '__main__':
    multiprocessing.set_start_method('fork')
    main(sys.argv[1:])
