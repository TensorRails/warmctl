"""The AMD side of the adapter, without a GPU: libkfdreplay's API as the
worker calls it, through a stand-in process that implements kfdreplay.h's
state machine, and the device's runtime names."""
import sys
import types
import unittest
from unittest import mock

from warmctl_vllm import device, kfdr, plugin
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


class PluginTest(unittest.TestCase):
    """Processes other than the workers checkpoint and restore their GPU
    state on the agent's signals."""

    def setUp(self):
        plugin._checkpointed = False
        self.addCleanup(setattr, plugin, '_checkpointed', False)

    def lib(self, **kw):
        return kfdr.Library(FakeKFDReplay(**kw))

    def test_cycle(self):
        lib = self.lib()
        self.assertEqual(plugin.checkpoint(lib), 'checkpointed')
        self.assertEqual(lib.state(), 'checkpointed_cold')
        self.assertEqual(plugin.restore(lib), 'restored')
        self.assertEqual(lib.state(), 'restored')

    def test_a_process_the_signal_did_not_checkpoint_skips(self):
        # A worker: the host restores it through the engine.
        lib = self.lib()
        lib.lib.state = 2
        self.assertEqual(plugin.restore(lib), 'skipped')
        self.assertEqual(lib.lib.calls, [])

    def test_refusal_and_failure_carry_the_reason(self):
        self.assertEqual(plugin.checkpoint(self.lib(refuse='REFUSE: drm')), 'error: REFUSE: drm')
        lib = self.lib(fail='queue 3 stalled')
        self.assertEqual(plugin.checkpoint(lib), 'checkpointed')
        self.assertEqual(plugin.restore(lib), 'error: queue 3 stalled; this process cannot use the GPU')

    def test_without_the_library_or_not_running(self):
        self.assertIn('not loaded', plugin.checkpoint(kfdr.Library(NoKFDReplay())))
        lib = self.lib()
        lib.lib.state = 3
        self.assertEqual(plugin.checkpoint(lib), 'error: libkfdreplay is restored, not running')

    def test_register_needs_the_status_directory(self):
        with mock.patch.dict('os.environ', {}, clear=True):
            self.assertFalse(plugin.register())

    @unittest.skipUnless(sys.platform.startswith('linux'), 'real-time signals are Linux\'s')
    def test_signals_report_to_the_status_directory(self):
        import os
        import signal
        import tempfile
        d = tempfile.mkdtemp()
        lib = FakeKFDReplay()
        with mock.patch.dict('os.environ', {plugin.STATUS_ENV: d}), \
                mock.patch.object(kfdr.ctypes, 'CDLL', lambda name: lib):
            old = signal.getsignal(plugin.CHECKPOINT_SIGNAL), signal.getsignal(plugin.RESTORE_SIGNAL)
            try:
                self.assertTrue(plugin.register())
                os.kill(os.getpid(), plugin.CHECKPOINT_SIGNAL)
                with open(os.path.join(d, str(os.getpid()))) as f:
                    self.assertEqual(f.read(), 'checkpointed\n')
                os.kill(os.getpid(), plugin.RESTORE_SIGNAL)
                with open(os.path.join(d, str(os.getpid()))) as f:
                    self.assertEqual(f.read(), 'restored\n')
            finally:
                signal.signal(plugin.CHECKPOINT_SIGNAL, old[0])
                signal.signal(plugin.RESTORE_SIGNAL, old[1])


class HostRegisterTest(unittest.TestCase):
    """vLLM 0.29's CudaRTLibrary on ROCm has no cudaHostRegister: the device
    binds the runtime's own; a wrapper that has it keeps being used."""

    def device(self, rt, hip):
        d = device.CudaDevice.__new__(device.CudaDevice)
        d.rt, d.hip = rt, hip
        return d

    def test_hip_pci_bus_id(self):
        def pci_bus_id(buf, size, index):
            self.assertEqual(index, 1)
            self.assertGreater(size, len(b'0000:c1:00.2'))
            buf.value = b'0000:c1:00.2'
            return 0

        lib = types.SimpleNamespace(hipDeviceGetPCIBusId=Fn(pci_bus_id))
        d = self.device(types.SimpleNamespace(lib=lib), True)
        d.index = 1
        self.assertEqual(d.hip_pci_bus_id(), '0000:c1:00.2')
        lib.hipDeviceGetPCIBusId = Fn(lambda *args: 42)
        with self.assertRaisesRegex(RuntimeError, 'hipDeviceGetPCIBusId'):
            d.hip_pci_bus_id()

    def test_falls_back_to_the_runtime(self):
        calls = []

        class Lib:
            pass
        lib = Lib()
        lib.hipHostRegister = Fn(lambda p, n, f: calls.append(('register', p, n, f)) or 0)
        lib.hipHostUnregister = Fn(lambda p: calls.append(('unregister', p)) or 0)
        lib.hipGetLastError = Fn(lambda: 0)
        rt = types.SimpleNamespace(lib=lib)
        d = self.device(rt, True)
        self.assertTrue(d.register(4096, 8192))
        d.unregister(4096)
        self.assertEqual(calls, [('register', 4096, 8192, 0), ('unregister', 4096)])

    def test_the_wrappers_own_when_it_has_them(self):
        calls = []
        rt = types.SimpleNamespace(lib=object(), cudaHostRegister=lambda p, n, f: calls.append('w') or 1,
                                   cudaGetLastError=lambda: calls.append('e'),
                                   cudaHostUnregister=lambda p: calls.append('u'))
        d = self.device(rt, False)
        self.assertFalse(d.register(1, 2))
        d.unregister(1)
        self.assertEqual(calls, ['w', 'e', 'u'])


class PlatformGuardTest(unittest.TestCase):
    def guard(self, hip):
        import platform

        from warmctl_vllm import worker
        torch = types.ModuleType('torch')
        torch.version = types.SimpleNamespace(hip=hip)
        calls = []

        def architecture(executable=sys.executable, bits='', linkage=''):
            calls.append(executable)
            return ('64bit', 'ELF')
        with mock.patch.dict(sys.modules, {'torch': torch}), mock.patch.object(platform, 'architecture', architecture):
            guarded = worker.rocm_platform_guard()
            first, second = platform.architecture(), platform.architecture()
            other = platform.architecture('/bin/sh')
        return guarded, calls, first, second, other

    def test_rocm_asks_once(self):
        guarded, calls, first, second, other = self.guard('7.16.26385')
        self.assertTrue(guarded)
        # Once at import (before the GPU runtime starts), then cached;
        # another executable is still asked.
        self.assertEqual(calls, [sys.executable, '/bin/sh'])
        self.assertEqual(first, ('64bit', 'ELF'))
        self.assertEqual(second, first)

    def test_cuda_untouched(self):
        guarded, calls, _, _, _ = self.guard(None)
        self.assertFalse(guarded)
        self.assertEqual(calls, [sys.executable, sys.executable, '/bin/sh'])


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
