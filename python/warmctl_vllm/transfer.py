"""Moving the weights pool between device memory and part files.

A `device` object does the CUDA work (device.py; a fake in the tests):
    bind()                      make the calling thread use the rank's GPU
    register(ptr, n)            page-lock host memory for DMA
    unregister(ptr)
    to_host(host_ptr, dev_ptr, n), to_device(dev_ptr, host_ptr, n)   synchronous copies
  and, when it can pipeline a reload (device.py does; WARMCTL_POOL_ASYNC=0
  turns it off):
    stream(), event()           a non-blocking stream of the calling thread, an event
    to_device_async(dev_ptr, host_ptr, n, stream), record(event, stream)
    wait(event), drain(stream), destroy(handle)
  and, to fingerprint as the data lands (WARMCTL_POOL_FUSED=0 turns it off):
    totals(rows)                a zeroed fingerprint table on the GPU
    accumulate(totals, row, ptr, n, position, stream)   add a region's fingerprint
    collect(tables, rows)       the tables' sum, after every stream drained

Host staging is anonymous mmap registered with the driver, one slot per
thread, released when done: after an export no byte of the weights stays in
the process's anonymous memory, so the CRIU dump does not carry them.
Threads work part by part; the copy, the SHA-256 and the file I/O of one
part overlap with those of the others (ctypes, hashlib and os all release
the GIL). A pipelined reload gives each thread two slots and a stream: it
reads chunk k+1 from the part file while chunk k is copied to the GPU.
"""
from __future__ import annotations

import ctypes
import hashlib
import mmap
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from . import pool
from .primitives import require

SLOT_BYTES = 64 << 20
THREADS = 4


def pipelined(device):
    """Whether a reload overlaps reads and copies on this device."""
    return hasattr(device, 'to_device_async') and os.environ.get('WARMCTL_POOL_ASYNC', '1') != '0'


def fused(device):
    """Whether a pipelined reload fingerprints each chunk as it lands."""
    return hasattr(device, 'accumulate') and os.environ.get('WARMCTL_POOL_FUSED', '1') != '0'


class Stats:
    """Seconds a reload spent, summed over its threads: reading part files
    into slots, and waiting for copies (a slot to come back, or the stream
    to drain at the end of a part)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.read = self.copy = 0.0
        # Set by a fused reload: every allocation's fingerprint, in order.
        self.sums = None

    def add(self, read, copy):
        with self.lock:
            self.read += read
            self.copy += copy


class Slot:
    def __init__(self, device, size):
        self.device, self.size = device, size
        self.region = mmap.mmap(-1, size)
        self.view = memoryview(self.region)
        self.ptr = ctypes.addressof(ctypes.c_char.from_buffer(self.region))
        self.registered = device.register(self.ptr, size)

    def close(self):
        if self.registered:
            self.device.unregister(self.ptr)
            self.registered = False
        self.view.release()
        self.region.close()


class Slots:
    """A pool of page-locked slots handed to threads one at a time."""

    def __init__(self, device, count, size):
        self.free = queue.Queue()
        self.all = []
        try:
            for _ in range(count):
                slot = Slot(device, size)
                self.all.append(slot)
                self.free.put(slot)
        except BaseException:
            self.close()
            raise

    def run(self, work):
        slot = self.free.get()
        try:
            return work(slot)
        finally:
            self.free.put(slot)

    def close(self):
        for slot in self.all:
            slot.close()
        self.all = []


def _parallel(device, threads, slot_bytes, jobs):
    slots = Slots(device, threads, slot_bytes)
    try:
        with ThreadPoolExecutor(max_workers=threads, initializer=device.bind) as executor:
            return list(executor.map(lambda job: slots.run(job), jobs))
    finally:
        slots.close()


def check_reads(device, allocation, position, nbytes=64 << 10):
    """The GPU fingerprint and a plain device-to-host copy must agree on an
    allocation's first bytes. A view over raw addresses that read the wrong
    memory would fingerprint garbage consistently on both sides; this
    catches it while baking instead of after a restore."""
    n = min(nbytes, allocation['bytes'])
    if n <= 0:
        return
    ptr = int(allocation['ptr'], 16)
    host = ctypes.create_string_buffer(n)
    device.to_host(ctypes.addressof(host), ptr, n)
    want = pool.reference_sum(host.raw, position)
    got = list(device.sums([(ptr, n, position)])[0])
    require(got == want, 'GPU fingerprint of %s disagrees with its host copy: %s vs %s'
            % (allocation['ptr'], got, want))


def _write_all(fd, view):
    while len(view):
        written = os.write(fd, view)
        view = view[written:]


def _read_exact(f, offset, view):
    f.seek(offset)
    while len(view):
        n = f.readinto(view)
        require(n, 'part file ends early')
        view = view[n:]


def export(device, allocations, parts, directory, rank, threads=THREADS, slot_bytes=SLOT_BYTES):
    """Write every part of the rank's stream to <directory>/rank-NNN/; return
    the part records (relative path, size, SHA-256)."""
    relative = 'rank-%03d' % rank
    os.makedirs(os.path.join(directory, relative))
    layout = pool.pieces(allocations, parts)

    def job(index):
        def work(slot):
            name = '%s/part-%05d.bin' % (relative, index)
            fd = os.open(os.path.join(directory, name), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            digest = hashlib.sha256()
            try:
                for allocation, inner, _offset, length in layout[index]:
                    source = int(allocations[allocation]['ptr'], 16) + inner
                    for done in range(0, length, slot.size):
                        n = min(slot.size, length - done)
                        device.to_host(slot.ptr, source + done, n)
                        digest.update(slot.view[:n])
                        _write_all(fd, slot.view[:n])
                os.fsync(fd)
            finally:
                os.close(fd)
            return {'index': index, 'path': name, 'size': parts[index]['size'], 'sha256': digest.hexdigest()}
        return work

    return _parallel(device, threads, slot_bytes, [job(part['index']) for part in parts])


class Lane:
    """Two page-locked slots, an event each and one stream, owned by one
    thread: a slot is filled while the other one's copy is in flight, and
    refilled only after its event fired."""

    def __init__(self, device, size, rows=0):
        self.device, self.size = device, size
        self.slots, self.events, self.pending = [], [], [False, False]
        self.stream = self.totals = None
        try:
            self.stream = device.stream()
            for _ in range(2):
                self.slots.append(Slot(device, size))
                self.events.append(device.event())
            if rows:
                self.totals = device.totals(rows)
        except BaseException:
            self.close()
            raise
        self.turn = 0
        self.read = self.copy = 0.0

    def fill(self, target, f, offset, n):
        """Read n bytes at offset of f into the next slot and queue its copy
        to target."""
        i, self.turn = self.turn, self.turn ^ 1
        if self.pending[i]:
            started = time.monotonic()
            self.device.wait(self.events[i])
            self.copy += time.monotonic() - started
        slot = self.slots[i]
        started = time.monotonic()
        _read_exact(f, offset, slot.view[:n])
        self.read += time.monotonic() - started
        self.device.to_device_async(target, slot.ptr, n, self.stream)
        self.device.record(self.events[i], self.stream)
        self.pending[i] = True

    def land(self, row, target, n, position):
        """Queue the fingerprint of the n bytes just copied to target behind
        the copy on this lane's stream: it reads them from device memory."""
        self.device.accumulate(self.totals, row, target, n, position, self.stream)

    def drain(self):
        """Wait for every queued copy: the reload is done only once its
        bytes are on the GPU."""
        started = time.monotonic()
        self.device.drain(self.stream)
        self.copy += time.monotonic() - started
        self.pending = [False, False]

    def close(self):
        # Copies still in flight read the slots: drain before unpinning.
        try:
            if self.stream is not None:
                try:
                    self.device.drain(self.stream)
                finally:
                    for event in self.events:
                        self.device.destroy(event)
                    self.device.destroy(self.stream)
                    self.stream = None
        finally:
            for slot in self.slots:
                slot.close()
            self.slots, self.events = [], []


def _pipelined(device, threads, slot_bytes, jobs, stats, rows=0):
    """Run job(lane) for every job on threads that each own a lane; the
    results come back in job order. With rows, each lane has a fingerprint
    table and stats.sums gets their sum."""
    tables = []
    work = queue.Queue()
    for item in enumerate(jobs):
        work.put(item)
    results = [None] * len(jobs)
    failed = threading.Event()

    def runner():
        device.bind()
        lane = Lane(device, slot_bytes, rows)
        if lane.totals is not None:
            with stats.lock:
                tables.append(lane.totals)
        try:
            while not failed.is_set():
                try:
                    index, job = work.get_nowait()
                except queue.Empty:
                    # The next part's reads overlap this part's last copies;
                    # the lane drains once, when the queue is empty.
                    lane.drain()
                    return
                results[index] = job(lane)
        except BaseException:
            failed.set()
            raise
        finally:
            try:
                lane.close()
            finally:
                stats.add(lane.read, lane.copy)

    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = [executor.submit(runner) for _ in range(max(1, min(threads, len(jobs))))]
    for future in futures:
        future.result()
    if rows:
        stats.sums = device.collect(tables, rows)
    return results


def reload(device, allocations, parts, directory, threads=THREADS, slot_bytes=SLOT_BYTES, verify=False, stats=None,
           pipeline=True):
    """Copy every listed allocation back from the part files; return the
    bytes of the parts read. With verify, each part is read whole and its
    SHA-256 checked against the record. stats, a Stats, collects where the
    threads spent their time."""
    layout = pool.pieces(allocations, parts)
    starts, _ = pool.part_starts(parts)
    stats = stats if stats is not None else Stats()
    if pipeline and not verify and pipelined(device):
        fuse = fused(device)

        def lane_job(index):
            def work(lane):
                part = parts[index]
                with open(os.path.join(directory, part['path']), 'rb', buffering=0) as f:
                    for allocation, inner, offset, length in layout[index]:
                        target = int(allocations[allocation]['ptr'], 16) + inner
                        for done in range(0, length, lane.size):
                            n = min(lane.size, length - done)
                            lane.fill(target + done, f, offset + done, n)
                            if fuse:
                                lane.land(allocation, target + done, n, starts[index] + offset + done)
                return part['size']
            return work

        jobs = [lane_job(part['index']) for part in parts]
        return sum(_pipelined(device, threads, slot_bytes, jobs, stats, len(allocations) if fuse else 0))

    def job(index):
        def work(slot):
            part = parts[index]
            with open(os.path.join(directory, part['path']), 'rb', buffering=0) as f:
                if verify:
                    _reload_verified(device, slot, f, part, layout[index], allocations)
                else:
                    _reload_ranges(device, slot, f, layout[index], allocations, stats)
            return part['size']
        return work

    return sum(_parallel(device, threads, slot_bytes, [job(part['index']) for part in parts]))


def _reload_ranges(device, slot, f, layout, allocations, stats):
    read = copy = 0.0
    for allocation, inner, offset, length in layout:
        target = int(allocations[allocation]['ptr'], 16) + inner
        for done in range(0, length, slot.size):
            n = min(slot.size, length - done)
            started = time.monotonic()
            _read_exact(f, offset + done, slot.view[:n])
            middle = time.monotonic()
            device.to_device(target + done, slot.ptr, n)
            read, copy = read + middle - started, copy + time.monotonic() - middle
    stats.add(read, copy)


def _reload_verified(device, slot, f, part, layout, allocations):
    digest = hashlib.sha256()
    for start in range(0, part['size'], slot.size):
        n = min(slot.size, part['size'] - start)
        _read_exact(f, start, slot.view[:n])
        digest.update(slot.view[:n])
        for allocation, inner, offset, length in layout:
            low, high = max(start, offset), min(start + n, offset + length)
            if low < high:
                target = int(allocations[allocation]['ptr'], 16) + inner + (low - offset)
                device.to_device(target, slot.ptr + (low - start), high - low)
    require(digest.hexdigest() == part['sha256'], 'part %s SHA-256 differs' % part['path'])
