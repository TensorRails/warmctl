"""libkfdreplay as an AMD worker calls it: the library the engine's processes
preload (LD_PRELOAD) to checkpoint and restore their own GPU state.

The library is not linked: its kfdr_* functions are looked up in the
process, as ctypes.CDLL(None) finds whatever the dynamic linker loaded."""
from __future__ import annotations

import ctypes

# enum kfdr_state of include/kfdreplay.h.
STATES = {0: 'running', 1: 'checkpointed_warm', 2: 'checkpointed_cold', 3: 'restored'}


class Library:
    """The preloaded libkfdreplay, or none (lib is None)."""

    def __init__(self, process=None):
        process = process if process is not None else ctypes.CDLL(None)
        if not hasattr(process, 'kfdr_checkpoint'):
            self.lib = None
            return
        self.lib = process
        for name, restype, argtypes in (
                ('kfdr_checkpoint', ctypes.c_int, [ctypes.c_int]),
                ('kfdr_restore', ctypes.c_int, []),
                ('kfdr_state', ctypes.c_int, []),
                ('kfdr_version', ctypes.c_char_p, []),
                ('kfdr_last_error', ctypes.c_char_p, []),
                ('kfdr_max_wptr', ctypes.c_uint64, [ctypes.c_void_p])):
            fn = getattr(process, name)
            fn.restype, fn.argtypes = restype, argtypes

    @property
    def loaded(self):
        return self.lib is not None

    def version(self):
        return _text(self.lib.kfdr_version()) if self.lib else ''

    def state(self):
        value = self.lib.kfdr_state()
        return STATES.get(value, 'unknown(%d)' % value)

    def last_error(self):
        return _text(self.lib.kfdr_last_error())

    def max_wptr(self):
        return int(self.lib.kfdr_max_wptr(None))

    def checkpoint(self):
        """A cold checkpoint: 0, or the library's reason for refusing."""
        return self.lib.kfdr_checkpoint(1)

    def restore(self):
        return self.lib.kfdr_restore()


def _text(raw):
    if raw is None:
        return ''
    return raw.decode(errors='replace') if isinstance(raw, bytes) else str(raw)
