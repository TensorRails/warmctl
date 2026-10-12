"""Moving the weights pool between device memory and part files.

A `device` object does the CUDA work (device.py; a fake in the tests):
    bind()                      make the calling thread use the rank's GPU
    register(ptr, n)            page-lock host memory for DMA
    unregister(ptr)
    to_host(host_ptr, dev_ptr, n), to_device(dev_ptr, host_ptr, n)   synchronous copies

Host staging is anonymous mmap registered with the driver, one slot per
thread, released when done: after an export no byte of the weights stays in
the process's anonymous memory, so the CRIU dump does not carry them.
Threads work part by part; the copy, the SHA-256 and the file I/O of one
part overlap with those of the others (ctypes, hashlib and os all release
the GIL).
"""
from __future__ import annotations

import ctypes
import hashlib
import mmap
import os
import queue
from concurrent.futures import ThreadPoolExecutor

from . import pool
from .primitives import require

SLOT_BYTES = 64 << 20
THREADS = 4


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


def reload(device, allocations, parts, directory, threads=THREADS, slot_bytes=SLOT_BYTES, verify=False):
    """Copy every listed allocation back from the part files. With verify,
    each part is read whole and its SHA-256 checked against the record."""
    layout = pool.pieces(allocations, parts)

    def job(index):
        def work(slot):
            part = parts[index]
            with open(os.path.join(directory, part['path']), 'rb', buffering=0) as f:
                if verify:
                    _reload_verified(device, slot, f, part, layout[index], allocations)
                else:
                    _reload_ranges(device, slot, f, layout[index], allocations)
            return part['size']
        return work

    return sum(_parallel(device, threads, slot_bytes, [job(part['index']) for part in parts]))


def _reload_ranges(device, slot, f, layout, allocations):
    for allocation, inner, offset, length in layout:
        target = int(allocations[allocation]['ptr'], 16) + inner
        for done in range(0, length, slot.size):
            n = min(slot.size, length - done)
            _read_exact(f, offset + done, slot.view[:n])
            device.to_device(target + done, slot.ptr, n)


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
