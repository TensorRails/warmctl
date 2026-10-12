"""Small pure helpers."""
from __future__ import annotations


def require(ok, message):
    if not ok:
        raise ValueError(message)


def rank_rows(rows, world):
    """The per-rank results of a collective RPC, checked and sorted by rank."""
    require(isinstance(rows, list) and len(rows) == world, 'incomplete worker list')
    rows = sorted(rows, key=lambda r: r['rank'])
    require([r['rank'] for r in rows] == list(range(world)), 'duplicate or missing rank')
    require(len({r['pid'] for r in rows}) == world, 'duplicate worker PID')
    return rows


def is_meta(tensor):
    """A meta tensor has a shape and dtype but no storage: vLLM leaves skipped
    submodules (a multimodal tower with limit-mm-per-prompt 0) on the meta
    device."""
    return getattr(getattr(tensor, 'device', None), 'type', None) == 'meta'
