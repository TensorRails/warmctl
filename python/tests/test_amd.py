"""The AMD side of the adapter, without a GPU: libkfdreplay's API as the
worker calls it, through a stand-in process that implements kfdreplay.h's
state machine, and the device's runtime names."""
import sys
import types
import unittest
from unittest import mock

from warmctl_vllm import device, kfdr
from warmctl_vllm.worker import Extension


class Fn:
    """A ctypes function stand-in: callable, with restype and argtypes."""

    def __init__(self, body):
        self.body, self.restype, self.argtypes = body, None, None

    def __call__(self, *args):
        return self.body(*args)


class FakeKFDReplay:
    """kfdreplay.h's behaviour: RUNNING → CHECKPOINTED_COLD → (restart) →
    RESTORED; refusal and failure leave a reason in kfdr_last_error."""

    def __init__(self, version='0.1.0', refuse='', fail=''):
        self.state, self.error = 0, b''
        self.refuse, self.fail, self.calls = refuse, fail, []
        self.kfdr_version = Fn(lambda: version.encode())
        self.kfdr_state = Fn(lambda: self.state)
        self.kfdr_last_error = Fn(lambda: self.error)
        self.kfdr_max_wptr = Fn(lambda slots: 41467)
        self.kfdr_checkpoint = Fn(self._checkpoint)
        self.kfdr_restore = Fn(self._restore)

    def _checkpoint(self, cold):
        self.calls.append(('checkpoint', cold))
        if self.refuse:
            self.error = self.refuse.encode()
            return -1
        self.state, self.error = 2 if cold else 1, b''
        return 0

    def _restore(self):
        self.calls.append(('restore',))
        if self.fail:
            self.error = self.fail.encode()
            return -1
        self.state, self.error = 3, b''
        return 0


class NoKFDReplay:
    """A process without the library: CDLL(None) has no kfdr_ symbols."""


class Rank(Extension):
    def __init__(self, rank=0):
        self.rank = rank

    def _warmctl_rank(self):
        return self.rank


def fake_torch():
    torch = types.ModuleType('torch')
    torch.cuda = types.SimpleNamespace(synchronize=lambda *a: None)
    return torch


class KFDReplayTest(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(sys.modules, {'torch': fake_torch()})
        patcher.start()
        self.addCleanup(patcher.stop)

    def use(self, process):
        patcher = mock.patch.object(kfdr.ctypes, 'CDLL', lambda name: process)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_cold_cycle(self):
        lib = FakeKFDReplay()
        self.use(lib)
        out = Rank(0).warmctl_gpu_checkpoint(version='0.1.0')
        self.assertEqual(lib.calls, [('checkpoint', 1)])
        self.assertEqual((out['state'], out['version'], out['max_wptr'], out['rank']),
                         ('checkpointed_cold', '0.1.0', 41467, 0))
        out = Rank(0).warmctl_gpu_restore(version='0.1.0')
        self.assertEqual(out['state'], 'restored')
        self.assertEqual(Rank(0).warmctl_gpu_state(),
                         {'rank': 0, 'loaded': True, 'version': '0.1.0', 'state': 'restored', 'last_error': ''})

    def test_refusal_carries_the_reason(self):
        self.use(FakeKFDReplay(refuse='REFUSE: 1 unknown KFD ioctl(s), last nr 0x07'))
        with self.assertRaisesRegex(RuntimeError, 'refused the checkpoint: REFUSE: 1 unknown KFD ioctl'):
            Rank().warmctl_gpu_checkpoint()

    def test_failed_restore_carries_the_reason(self):
        lib = FakeKFDReplay(fail='queue 3: fast-forward stalled')
        lib.state = 2
        self.use(lib)
        with self.assertRaisesRegex(RuntimeError, 'fast-forward stalled; this process cannot use the GPU'):
            Rank().warmctl_gpu_restore()

    def test_without_the_library(self):
        self.use(NoKFDReplay())
        with self.assertRaisesRegex(Exception, 'not loaded in this process'):
            Rank().warmctl_gpu_checkpoint()
        self.assertEqual(Rank().warmctl_gpu_state(), {'rank': 0, 'loaded': False})
        self.assertEqual(kfdr.Library().version(), '')

    def test_version_mismatch(self):
        self.use(FakeKFDReplay(version='0.2.0'))
        with self.assertRaisesRegex(Exception, 'runs libkfdreplay 0.2.0, warmctl expects 0.1.0'):
            Rank().warmctl_gpu_checkpoint(version='0.1.0')

    def test_a_restored_engine_is_not_checkpointed_again(self):
        lib = FakeKFDReplay()
        lib.state = 3
        self.use(lib)
        with self.assertRaisesRegex(Exception, 'freshly started engine'):
            Rank().warmctl_gpu_checkpoint()
        self.assertEqual(lib.calls, [])

    def test_restore_needs_a_checkpoint(self):
        self.use(FakeKFDReplay())
        with self.assertRaisesRegex(Exception, 'running, not checkpointed'):
            Rank().warmctl_gpu_restore()

    def test_prepare_without_vllms_hook(self):
        class Bare(Rank):
            def _warmctl_collectives(self):
                return []
        self.assertEqual(Bare().warmctl_checkpoint_prepare(),
                         {'rank': 0, 'collectives': [], 'detached': True, 'nccl_mode': 'suspend'})
        self.assertEqual(Bare().warmctl_checkpoint_restore(), {'rank': 0})

    def test_prepare_with_vllms_hook(self):
        called = []

        class Hooked(Rank):
            def checkpoint_prepare(self):
                called.append('prepare')

            def checkpoint_restore(self):
                called.append('restore')

            def _warmctl_collectives(self):
                return []
        Hooked().warmctl_checkpoint_prepare()
        Hooked().warmctl_checkpoint_restore()
        self.assertEqual(called, ['prepare', 'restore'])


class RuntimeNameTest(unittest.TestCase):
    def test_cuda_names_stay(self):
        for name in ('cudaStreamCreateWithFlags', 'cudaMemcpyAsync', 'cudaEventDestroy'):
            self.assertEqual(device.runtime_name(name, False), name)

    def test_hip_names(self):
        self.assertEqual(device.runtime_name('cudaStreamCreateWithFlags', True), 'hipStreamCreateWithFlags')
        self.assertEqual(device.runtime_name('cudaEventSynchronize', True), 'hipEventSynchronize')

    def test_binding_takes_the_runtimes_names(self):
        class Lib:
            pass
        for hip, prefix in ((False, 'cuda'), (True, 'hip')):
            lib = Lib()
            for name in ('StreamCreateWithFlags', 'EventCreateWithFlags', 'MemcpyAsync', 'EventRecord',
                         'EventSynchronize', 'StreamSynchronize', 'StreamDestroy', 'EventDestroy'):
                setattr(lib, prefix + name, Fn(lambda *a: 0))
            bound = device._runtime(lib, hip)
            self.assertEqual(sorted(bound), sorted('cuda' + n for n in (
                'StreamCreateWithFlags', 'EventCreateWithFlags', 'MemcpyAsync', 'EventRecord', 'EventSynchronize',
                'StreamSynchronize', 'StreamDestroy', 'EventDestroy')))


if __name__ == '__main__':
    unittest.main()
