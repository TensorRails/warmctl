"""warmctl's vLLM general plugin (entry point vllm.general_plugins), loaded by
vLLM in every process of the engine: the API server, the engine core and the
workers.

On AMD, processes other than the workers initialize ROCm too: vLLM 0.33's
API server and engine core import kernels that probe the GPU (aiter, Triton's
AMD target, flash-linear-attention). Their runtime then holds the devices,
which a snapshot must release. The workers checkpoint their GPU state when the
host asks them through the engine (worker.py); this plugin lets every other
process do the same on a signal from warmctl's agent:

  CHECKPOINT_SIGNAL   kfdr_checkpoint(1), then "checkpointed"
  RESTORE_SIGNAL      kfdr_restore() in a process the signal checkpointed,
                      then "restored"; any other process answers "skipped"

each written to <$WARMCTL_KFDR_STATUS>/<pid> ("error: <reason>" on failure).
The constants match internal/rocm/signal.go. Without $WARMCTL_KFDR_STATUS,
which warmctl sets only for AMD engines, the plugin does nothing."""
from __future__ import annotations

import os
import signal
import sys

from . import kfdr

# SIGRTMIN+6 and SIGRTMIN+7 of glibc.
CHECKPOINT_SIGNAL = 40
RESTORE_SIGNAL = 41
STATUS_ENV = 'WARMCTL_KFDR_STATUS'

_checkpointed = False


def register():
    """vLLM calls this once per process, before the engine starts."""
    if not os.environ.get(STATUS_ENV):
        return False
    try:
        signal.signal(CHECKPOINT_SIGNAL, _on_checkpoint)
        signal.signal(RESTORE_SIGNAL, _on_restore)
    except ValueError:
        # Not the main thread: this process cannot be asked.
        return False
    return True


def _report(state):
    directory = os.environ.get(STATUS_ENV, '')
    name = os.path.join(directory, str(os.getpid()))
    try:
        os.makedirs(directory, exist_ok=True)
        with open(name + '.tmp', 'w') as f:
            f.write(state + '\n')
        os.replace(name + '.tmp', name)
    except OSError as e:
        sys.stderr.write('warmctl: cannot report GPU state %r to %s: %s\n' % (state, name, e))


def checkpoint(lib=None):
    """Cold checkpoint of this process's GPU state; the status to report."""
    global _checkpointed
    lib = lib if lib is not None else kfdr.Library()
    if not lib.loaded:
        return 'error: libkfdreplay is not loaded in process %d' % os.getpid()
    state = lib.state()
    if state != 'running':
        return 'error: libkfdreplay is %s, not running' % state
    if lib.checkpoint() != 0:
        return 'error: ' + (lib.last_error() or 'refused, no reason given')
    _checkpointed = True
    return 'checkpointed'


def restore(lib=None):
    """Replay this process's GPU state if the signal checkpointed it."""
    global _checkpointed
    if not _checkpointed:
        return 'skipped'
    lib = lib if lib is not None else kfdr.Library()
    if lib.restore() != 0:
        return 'error: ' + (lib.last_error() or 'no reason given') + '; this process cannot use the GPU'
    _checkpointed = False
    return 'restored'


def _on_checkpoint(signum, frame):
    _report(checkpoint())


def _on_restore(signum, frame):
    _report(restore())
