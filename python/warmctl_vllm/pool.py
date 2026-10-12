"""The weights pool, as data.

vLLM's sleep-mode allocator (CuMemAllocator) records every allocation it
made as {ptr: AllocationData(handle=(_, size), tag, is_asleep, ...)}. The
ones tagged `weights` are the processed model: parameters, buffers,
quantization scales, derived caches, the drafter, scratch. A snapshot
exports all of them as bytes in address order, one stream cut into parts of
at most MAX_PART_BYTES; a restore maps the pool back (`wake_up(tags=
["weights"])` gives the same addresses) and copies the parts in. No tensor
is named anywhere.

Correctness is a fingerprint per allocation: the sum, the sum of squares
and the position-weighted sum (weight (position % 251) + 1, position counted
in the pool stream) of its bytes, read from device memory on both sides.
Everything here is pure; transfer.py moves bytes, device.py touches CUDA.
"""
from __future__ import annotations

from .primitives import require

TAG = 'weights'
MAX_PART_BYTES = 1 << 30


def allocation_rows(pointer_to_data):
    """The allocator table as plain rows sorted by address."""
    rows = []
    for ptr, data in pointer_to_data.items():
        rows.append({'ptr': hex(int(ptr)), 'bytes': int(data.handle[1]), 'tag': str(data.tag),
                     'asleep': bool(data.is_asleep)})
    rows.sort(key=lambda row: int(row['ptr'], 16))
    return rows


def weights_rows(rows):
    return [row for row in rows if row['tag'] == TAG]


def layout(rows):
    """(ptr, bytes, tag) of every weights allocation: what must match after wake."""
    return [(row['ptr'], row['bytes'], row['tag']) for row in weights_rows(rows)]


def plan_parts(rows, part_bytes=MAX_PART_BYTES):
    """Lay the weights allocations out as one stream cut into parts.

    Returns (allocations, parts): each allocation gets index, ptr, bytes, tag
    and the part and offset of its first byte (it may continue into the next
    parts); parts carry index and size."""
    require(0 < part_bytes <= MAX_PART_BYTES, 'part size must be within 1 GiB')
    allocations, total = [], 0
    for index, row in enumerate(weights_rows(rows)):
        require(row['bytes'] >= 0, 'negative allocation')
        allocations.append({'index': index, 'ptr': row['ptr'], 'bytes': row['bytes'], 'tag': row['tag'],
                            'part': total // part_bytes, 'offset': total % part_bytes})
        total += row['bytes']
    parts = [{'index': i, 'size': min(part_bytes, total - i * part_bytes)} for i in range(-(-total // part_bytes))]
    return allocations, parts


def part_starts(parts):
    starts, total = [], 0
    for part in parts:
        starts.append(total)
        total += part['size']
    return starts, total


def stream_start(allocation, starts):
    return starts[allocation['part']] + allocation['offset']


def pieces(allocations, parts):
    """For every part, the pieces it holds as (allocation index, offset in the
    allocation, offset in the part, length), in part order. Allocations are
    placed by their own part and offset, so a manifest that dropped some
    allocations leaves holes, not shifts."""
    starts, total = part_starts(parts)
    out = [[] for _ in parts]
    for position, allocation in enumerate(allocations):
        start = stream_start(allocation, starts)
        require(start + allocation['bytes'] <= total, 'allocation %d runs past the pool stream' % position)
        inner, at = 0, start
        while inner < allocation['bytes']:
            part = _part_at(starts, at)
            offset = at - starts[part]
            length = min(allocation['bytes'] - inner, parts[part]['size'] - offset)
            out[part].append((position, inner, offset, length))
            inner += length
            at += length
    for row in out:
        row.sort(key=lambda piece: piece[2])
    return out


def _part_at(starts, at):
    low, high = 0, len(starts) - 1
    while low < high:
        mid = (low + high + 1) // 2
        if starts[mid] <= at:
            low = mid
        else:
            high = mid - 1
    return low


def reference_sum(data, position=0):
    """The fingerprint of one allocation's bytes starting at stream position.
    The test oracle for device.py; exact integer arithmetic."""
    s = sq = w = 0
    for i, b in enumerate(data):
        s += b
        sq += b * b
        w += b * (((position + i) % 251) + 1)
    return [s, sq, w]


def total(sums):
    out = [0, 0, 0]
    for row in sums:
        out = [a + b for a, b in zip(out, row)]
    return out


def compare(expected, actual):
    """Indices of allocations whose fingerprints differ."""
    require(len(expected) == len(actual), 'fingerprint count differs: %d vs %d' % (len(expected), len(actual)))
    return [i for i, (a, b) in enumerate(zip(expected, actual)) if list(a) != list(b)]
