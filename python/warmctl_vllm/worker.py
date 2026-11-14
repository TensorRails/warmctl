"""The worker extension vLLM mixes into its Worker (--worker-extension-cls).

Every method is one RPC the host calls through /collective_rpc. A method
returns a plain JSON value or raises; vLLM carries the exception back.
"""
from __future__ import annotations

import json
import os
import time

from . import kfdr, pool, protocol, transfer
from .primitives import require


def replica(config):
    """The data-parallel replica of a worker. vLLM sets data_parallel_rank to
    0 in every replica of a dense model (no expert parallelism: the replicas
    share no process group) and keeps the replica in data_parallel_index;
    versions before that field number the replicas in data_parallel_rank."""
    index = getattr(config, 'data_parallel_index', None)
    return config.data_parallel_rank if index is None else index

class Extension:
    def warmctl_rpc(self, method, kwargs, reply):
        """Run method, warmctl's or vLLM's own worker method, and publish its
        result as <reply>/<rank>.json. With data parallelism /collective_rpc
        runs on every engine but answers with the first engine's workers only;
        the host reads every rank's file instead."""
        protocol.operation(method)
        protocol.absolute(reply)
        result = protocol.plain(getattr(self, method)(**kwargs))
        os.makedirs(reply, exist_ok=True)
        protocol.publish('%s/%d.json' % (reply, self._warmctl_rank()), {'result': result})
        return result

    def warmctl_describe(self):
        import torch
        import vllm

        props = torch.cuda.get_device_properties(self.device)
        bus = getattr(props, 'pci_bus_id', None)
        row = {
            'rank': self._warmctl_rank(), 'pid': os.getpid(), 'device': torch.cuda.current_device(),
            'name': props.name, 'arch': 'sm_%d%d' % (props.major, props.minor),
            'pci_bus_id': '%04x:%02x:%02x.0' % (getattr(props, 'pci_domain_id', 0), bus, props.pci_device_id)
            if bus is not None else '',
            'uuid': str(getattr(props, 'uuid', '')), 'vllm': vllm.__version__, 'torch': torch.__version__,
            'adapter_sha256': os.environ.get('WARMCTL_ADAPTER_SHA256', ''),
        }
        hip = getattr(torch.version, 'hip', None)
        if hip:
            # ROCm: the ISA by its gfx name, and what moves the GPU state.
            row['arch'] = str(getattr(props, 'gcnArchName', row['arch'])).split(':')[0]
            row['hip'], row['rocm'] = str(hip), kfdr_runtime()
            row['kfdr_version'] = kfdr.Library().version()
        return protocol.plain(row)

    def warmctl_allocations(self):
        """The allocator table: what the pool holds right now."""
        return {'rank': self._warmctl_rank(), 'allocations': pool.allocation_rows(self._warmctl_table())}

    def warmctl_export_pool(self, directory, part_bytes=pool.MAX_PART_BYTES, threads=transfer.THREADS,
                            slot_bytes=transfer.SLOT_BYTES):
        """Fingerprint and write the awake weights pool to directory. The engine
        must be idle: nothing may write the pool while it is read."""
        started = time.monotonic()
        rows = [r for r in pool.allocation_rows(self._warmctl_table()) if not r['asleep']]
        allocations, parts = pool.plan_parts(rows, part_bytes)
        device = self._warmctl_device()
        starts, _ = pool.part_starts(parts)
        sums = device.sums([(int(a['ptr'], 16), a['bytes'], pool.stream_start(a, starts)) for a in allocations])
        for allocation, row in zip(allocations, sums):
            allocation['sum'] = row
        if allocations:
            transfer.check_reads(device, allocations[0], pool.stream_start(allocations[0], starts))
        written = transfer.export(device, allocations, parts, directory, self._warmctl_rank(), threads, slot_bytes)
        return {'rank': self._warmctl_rank(), 'pid': os.getpid(), 'allocations': allocations, 'parts': written,
                'seconds': time.monotonic() - started}

    def warmctl_sleep_prepare(self):
        """After sleep(2) and vLLM's checkpoint_prepare: report whether this
        rank's CUDA restore depends on any other process."""
        active = self._warmctl_collectives()
        return {'rank': self._warmctl_rank(), 'collectives': active, 'detached': not active,
                'nccl_mode': 'suspend'}

    def warmctl_restore_prepare(self):
        return {'rank': self._warmctl_rank()}

    # AMD: GPU state moves in the worker itself, through libkfdreplay. The
    # NVIDIA path never calls these; its GPU state moves in warmctl's agent.

    def warmctl_checkpoint_prepare(self):
        """vLLM's checkpoint_prepare where this vLLM has one (the ROCm images
        may predate it), then warmctl's."""
        prepare = getattr(self, 'checkpoint_prepare', None)
        if prepare is not None:
            prepare()
        return self.warmctl_sleep_prepare()

    def warmctl_checkpoint_restore(self):
        restore = getattr(self, 'checkpoint_restore', None)
        if restore is not None:
            restore()
        return self.warmctl_restore_prepare()

    def warmctl_gpu_checkpoint(self, version=''):
        """Cold checkpoint of this process's GPU state, on the sleeping engine:
        GPU memory dumped to host memory, the runtime's threads parked, the
        devices released, so that DMTCP can snapshot the process."""
        import torch

        lib = self._warmctl_kfdr(version)
        state = lib.state()
        require(state != 'restored', 'this engine was restored from a snapshot; a snapshot of it is not '
                'supported (take one of a freshly started engine)')
        require(state == 'running', 'libkfdreplay is %s, not running' % state)
        torch.cuda.synchronize()
        max_wptr = lib.max_wptr()
        started = time.monotonic()
        if lib.checkpoint() != 0:
            raise RuntimeError('libkfdreplay refused the checkpoint: %s' % (lib.last_error() or 'no reason given'))
        state = lib.state()
        require(state == 'checkpointed_cold', 'libkfdreplay is %s after the checkpoint' % state)
        return {'rank': self._warmctl_rank(), 'pid': os.getpid(), 'version': lib.version(), 'state': state,
                'seconds': time.monotonic() - started, 'max_wptr': max_wptr}

    def warmctl_gpu_restore(self, version=''):
        """Replay this restored process's GPU state, before anything uses the
        GPU (the engine is still asleep)."""
        lib = self._warmctl_kfdr(version)
        state = lib.state()
        require(state == 'checkpointed_cold', 'libkfdreplay is %s, not checkpointed' % state)
        started = time.monotonic()
        if lib.restore() != 0:
            raise RuntimeError('libkfdreplay could not restore the GPU state: %s; this process cannot use the '
                               'GPU' % (lib.last_error() or 'no reason given'))
        state = lib.state()
        require(state == 'restored', 'libkfdreplay is %s after the restore' % state)
        return {'rank': self._warmctl_rank(), 'pid': os.getpid(), 'version': lib.version(), 'state': state,
                'seconds': time.monotonic() - started}

    def warmctl_gpu_state(self):
        lib = kfdr.Library()
        if not lib.loaded:
            return {'rank': self._warmctl_rank(), 'loaded': False}
        return {'rank': self._warmctl_rank(), 'loaded': True, 'version': lib.version(), 'state': lib.state(),
                'last_error': lib.last_error()}

    def _warmctl_kfdr(self, version):
        lib = kfdr.Library()
        require(lib.loaded, 'libkfdreplay is not loaded in this process: the engine was not started with it '
                'preloaded (LD_PRELOAD)')
        require(not version or lib.version() == version,
                'this process runs libkfdreplay %s, warmctl expects %s' % (lib.version(), version))
        return lib

    def warmctl_reload_pool(self, manifest, directory, threads=transfer.THREADS, slot_bytes=transfer.SLOT_BYTES,
                            verify=False, sync=False, wait=False):
        """Copy the rank's pool back from the snapshot after wake_up(weights)
        and check every allocation's fingerprint, before any forward pass.
        With wait, a part's file may still be on its way."""
        started = time.monotonic()
        rank = self._warmctl_rank()
        with open(manifest, 'rb') as f:
            record = json.load(f)['weights']['ranks'][rank]
        require(record['rank'] == rank, 'manifest rank %d at position %d' % (record['rank'], rank))
        live = {(r['ptr'], r['bytes'], r['tag']) for r in pool.allocation_rows(self._warmctl_table())
                if not r['asleep']}
        missing = [a['ptr'] for a in record['allocations'] if (a['ptr'], a['bytes'], a['tag']) not in live]
        require(not missing, 'weights pool layout differs after wake: %d allocations missing, first %s'
                % (len(missing), missing[:3]))
        device = self._warmctl_device()
        stats = transfer.Stats()
        copied = time.monotonic()
        loaded = transfer.reload(device, record['allocations'], record['parts'], directory, threads, slot_bytes,
                                 verify, stats, pipeline=not sync, wait=wait)
        summed = time.monotonic()
        sums = stats.sums
        if sums is None:
            starts, _ = pool.part_starts(record['parts'])
            sums = device.sums([(int(a['ptr'], 16), a['bytes'], pool.stream_start(a, starts))
                                for a in record['allocations']])
        done = time.monotonic()
        bad = pool.compare([a['sum'] for a in record['allocations']], sums)
        require(not bad, 'weights fingerprint differs in %d allocations, first %s'
                % (len(bad), [record['allocations'][i]['ptr'] for i in bad[:3]]))
        return {'rank': rank, 'bytes': loaded, 'allocations': len(sums), 'seconds': done - started,
                'reload_seconds': summed - copied, 'fingerprint_seconds': done - summed,
                'read_seconds': stats.read, 'copy_seconds': stats.copy, 'threads': threads,
                'pipelined': not verify and not sync and transfer.pipelined(device),
                'fused': stats.sums is not None}

    def _warmctl_rank(self):
        from vllm.distributed import parallel_state

        tp = parallel_state.get_tensor_model_parallel_rank()
        return replica(self.vllm_config.parallel_config) * self.vllm_config.parallel_config.tensor_parallel_size + tp

    def _warmctl_table(self):
        from vllm.device_allocator.cumem import CuMemAllocator

        return CuMemAllocator.get_instance().pointer_to_data

    def _warmctl_device(self):
        import torch

        from .device import CudaDevice

        return CudaDevice(torch, self.device)

    def _warmctl_collectives(self):
        """Collectives that keep peer memory mapped across processes: custom,
        quick, symmetric-memory and FlashInfer all-reduce. Without them a
        rank's CUDA state is its own and the ranks restore in parallel."""
        from vllm.distributed import parallel_state

        active = set()
        for reference in parallel_state._groups.values():
            group = reference()
            communicator = getattr(group, 'device_communicator', None)
            for name in ('ca_comm', 'qr_comm', 'symm_mem_comm', 'fi_ar_comm', 'fi_pcie_ipc_ar_comm'):
                value = getattr(communicator, name, None)
                if value is not None and not getattr(value, 'disabled', False):
                    active.add(name)
        return sorted(active)


def kfdr_runtime():
    """The ROCm userspace's version: the TheRock wheels' or /opt/rocm's."""
    try:
        from importlib import metadata
        for name in ('rocm-sdk-core', 'rocm_sdk_core', '_rocm_sdk_core'):
            try:
                return 'rocm-sdk-core ' + metadata.version(name)
            except metadata.PackageNotFoundError:
                continue
    except ImportError:
        pass
    try:
        with open('/opt/rocm/.info/version') as f:
            return 'rocm ' + f.read().strip()
    except OSError:
        return ''
