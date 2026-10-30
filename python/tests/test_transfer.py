import os
import random
import tempfile
import threading
import unittest

from warmctl_vllm import pool, transfer


class FakeDevice:
    """Device memory as one bytearray at a fake base address."""

    def __init__(self, base, size):
        self.base, self.memory = base, bytearray(size)
        self.registered, self.bound = set(), set()
        self.lock = threading.Lock()

    def bind(self):
        self.bound.add(threading.get_ident())

    def register(self, ptr, n):
        self.registered.add(ptr)
        return True

    def unregister(self, ptr):
        self.registered.discard(ptr)

    def _host(self, ptr, n):
        import ctypes
        return (ctypes.c_char * n).from_address(ptr)

    def to_host(self, host_ptr, dev_ptr, n):
        assert threading.get_ident() in self.bound
        self._host(host_ptr, n)[:] = bytes(self.memory[dev_ptr - self.base:dev_ptr - self.base + n])

    def to_device(self, dev_ptr, host_ptr, n):
        assert threading.get_ident() in self.bound
        self.memory[dev_ptr - self.base:dev_ptr - self.base + n] = bytes(self._host(host_ptr, n)[:n])

    def sums(self, regions):
        return [pool.reference_sum(self.memory[p - self.base:p - self.base + n], pos) for p, n, pos in regions]


class TransferTest(unittest.TestCase):
    def setUp(self):
        rnd = random.Random(3)
        self.base = 0x7f0000000000
        self.device = FakeDevice(self.base, 1 << 16)
        self.device.memory[:] = bytes(rnd.getrandbits(8) for _ in range(1 << 16))
        # Three allocations with gaps between them.
        self.rows = [{'ptr': hex(self.base + off), 'bytes': n, 'tag': 'weights', 'asleep': False}
                     for off, n in ((0, 5000), (8192, 20000), (40000, 7000))]

    def test_export_reload_roundtrip(self):
        allocations, parts = pool.plan_parts(self.rows, part_bytes=4096)
        before = bytes(self.device.memory)
        with tempfile.TemporaryDirectory() as root:
            written = transfer.export(self.device, allocations, parts, root, rank=3, threads=3, slot_bytes=1000)
            self.assertEqual([p['index'] for p in written], list(range(len(parts))))
            self.assertEqual(sum(p['size'] for p in written), 32000)
            self.assertTrue(all(os.path.getsize(os.path.join(root, p['path'])) == p['size'] for p in written))
            self.assertFalse(self.device.registered)
            # Wipe the pool, reload everything, then reload with verification.
            for a in allocations:
                start = int(a['ptr'], 16) - self.base
                self.device.memory[start:start + a['bytes']] = bytes(a['bytes'])
            self.assertEqual(transfer.reload(self.device, allocations, written, root, threads=2, slot_bytes=777), 32000)
            self.assertEqual(bytes(self.device.memory), before)
            transfer.reload(self.device, allocations, written, root, threads=2, slot_bytes=1000, verify=True)
            self.assertEqual(bytes(self.device.memory), before)
            # Dropping an allocation leaves its bytes alone.
            middle = int(allocations[1]['ptr'], 16) - self.base
            self.device.memory[middle] ^= 0xFF
            transfer.reload(self.device, [allocations[0], allocations[2]], written, root, threads=2, slot_bytes=1000)
            self.assertNotEqual(bytes(self.device.memory), before)
            # A corrupted part fails verification.
            with open(os.path.join(root, written[2]['path']), 'r+b') as f:
                f.seek(10)
                f.write(b'\x00')
            with self.assertRaises(ValueError):
                transfer.reload(self.device, allocations, written, root, threads=1, slot_bytes=1000, verify=True)

    def test_export_refuses_existing_rank(self):
        allocations, parts = pool.plan_parts(self.rows, part_bytes=4096)
        with tempfile.TemporaryDirectory() as root:
            transfer.export(self.device, allocations, parts, root, rank=0, threads=1, slot_bytes=4096)
            with self.assertRaises(FileExistsError):
                transfer.export(self.device, allocations, parts, root, rank=0, threads=1, slot_bytes=4096)


class AsyncFakeDevice(FakeDevice):
    """Copies queued on a stream run only when an event or the stream is
    waited on, reading the host slot then, as a DMA engine would: a reload
    that refills a slot before its copy finished loads the wrong bytes."""

    def __init__(self, base, size, fail_after=None):
        super().__init__(base, size)
        self.queues, self.done, self.handles = {}, {}, set()
        self.fail_after, self.copies = fail_after, 0

    def stream(self):
        handle = ('stream', object())
        self.queues[handle] = []
        self.handles.add(handle)
        return handle

    def event(self):
        handle = ('event', object())
        self.done[handle] = None
        self.handles.add(handle)
        return handle

    def to_device_async(self, dev_ptr, host_ptr, n, stream):
        assert threading.get_ident() in self.bound
        self.copies += 1
        if self.fail_after is not None and self.copies > self.fail_after:
            raise RuntimeError('cudaMemcpyAsync: injected')
        self.queues[stream].append(('copy', dev_ptr, host_ptr, n))

    def record(self, event, stream):
        self.queues[stream].append(('event', event))
        self.done[event] = stream

    def _run(self, stream, until=None):
        while self.queues[stream]:
            op = self.queues[stream].pop(0)
            if op[0] == 'copy':
                self.to_device(*op[1:])
            elif op[1] == until:
                return

    def wait(self, event):
        self._run(self.done[event], until=event)

    def drain(self, stream):
        self._run(stream)

    def destroy(self, handle):
        self.handles.remove(handle)


class PipelinedReloadTest(unittest.TestCase):
    def setUp(self):
        rnd = random.Random(11)
        self.base = 0x7f0000000000
        self.device = AsyncFakeDevice(self.base, 1 << 16)
        self.device.memory[:] = bytes(rnd.getrandbits(8) for _ in range(1 << 16))
        self.rows = [{'ptr': hex(self.base + off), 'bytes': n, 'tag': 'weights', 'asleep': False}
                     for off, n in ((0, 5000), (8192, 20000), (40000, 7000))]

    def _snapshot(self, root):
        allocations, parts = pool.plan_parts(self.rows, part_bytes=4096)
        sync = FakeDevice(self.base, 1 << 16)
        sync.memory[:] = self.device.memory
        written = transfer.export(sync, allocations, parts, root, rank=0, threads=2, slot_bytes=1000)
        for a in allocations:
            start = int(a['ptr'], 16) - self.base
            self.device.memory[start:start + a['bytes']] = bytes(a['bytes'])
        return allocations, written, bytes(sync.memory)

    def test_pipelined_reload_matches(self):
        with tempfile.TemporaryDirectory() as root:
            allocations, written, before = self._snapshot(root)
            self.assertTrue(transfer.pipelined(self.device))
            stats = transfer.Stats()
            # Small slots: every part takes several chunks, so both slots of a
            # lane are refilled while the other one's copy is still queued.
            self.assertEqual(transfer.reload(self.device, allocations, written, root, threads=3, slot_bytes=333,
                                             stats=stats), 32000)
            self.assertEqual(bytes(self.device.memory), before)
            self.assertFalse(self.device.registered)
            self.assertFalse(self.device.handles)
            self.assertGreater(stats.read, 0)

    def test_switch_off(self):
        with tempfile.TemporaryDirectory() as root:
            allocations, written, before = self._snapshot(root)
            os.environ['WARMCTL_POOL_ASYNC'] = '0'
            try:
                self.assertFalse(transfer.pipelined(self.device))
                transfer.reload(self.device, allocations, written, root, threads=2, slot_bytes=500)
            finally:
                del os.environ['WARMCTL_POOL_ASYNC']
            self.assertEqual(bytes(self.device.memory), before)
            self.assertFalse(self.device.copies)

    def test_failed_copy_releases_everything(self):
        with tempfile.TemporaryDirectory() as root:
            allocations, written, _ = self._snapshot(root)
            self.device.fail_after = 7
            with self.assertRaises(RuntimeError):
                transfer.reload(self.device, allocations, written, root, threads=3, slot_bytes=333)
            self.assertFalse(self.device.registered)
            self.assertFalse(self.device.handles)


if __name__ == '__main__':
    unittest.main()


class CheckReadsTest(unittest.TestCase):
    def setUp(self):
        self.base = 0x7f0000000000
        self.device = FakeDevice(self.base, 1 << 16)
        rnd = random.Random(5)
        self.device.memory[:] = bytes(rnd.getrandbits(8) for _ in range(1 << 16))
        self.device.bind()
        self.allocation = {'ptr': hex(self.base + 4096), 'bytes': 20000}

    def test_agreeing_paths_pass(self):
        transfer.check_reads(self.device, self.allocation, 777)
        transfer.check_reads(self.device, {'ptr': hex(self.base), 'bytes': 0}, 0)

    def test_view_reading_other_memory_fails(self):
        class Shifted(FakeDevice):
            def sums(self, regions):
                return super().sums([(p + 8, n, pos) for p, n, pos in regions])
        shifted = Shifted(self.base, 1 << 16)
        shifted.memory[:] = self.device.memory
        shifted.bind()
        with self.assertRaises(ValueError):
            transfer.check_reads(shifted, self.allocation, 0)
