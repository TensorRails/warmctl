"""The CUDA side of the pool: copies through vLLM's own CUDA runtime wrapper
(the one CuMemAllocator itself uses for sleep backups) and fingerprints
computed on the GPU over raw allocation addresses."""
from __future__ import annotations

import ctypes

BLOCK_BYTES = 8 << 20

# From driver_types.h.
_STREAM_NON_BLOCKING = 1
_EVENT_DISABLE_TIMING = 2
_MEMCPY_HOST_TO_DEVICE = 1

_bound = {}


def _runtime(lib):
    """Stream, event and async-copy functions of a libcudart, bound once."""
    if id(lib) not in _bound:
        handle, size, kind, err = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int
        signatures = {
            'cudaStreamCreateWithFlags': [ctypes.POINTER(handle), ctypes.c_uint],
            'cudaEventCreateWithFlags': [ctypes.POINTER(handle), ctypes.c_uint],
            'cudaMemcpyAsync': [handle, handle, size, kind, handle],
            'cudaEventRecord': [handle, handle],
            'cudaEventSynchronize': [handle],
            'cudaStreamSynchronize': [handle],
            'cudaStreamDestroy': [handle],
            'cudaEventDestroy': [handle],
        }
        functions = {}
        for name, argtypes in signatures.items():
            fn = getattr(lib, name)
            fn.restype, fn.argtypes = err, argtypes
            functions[name] = fn
        _bound[id(lib)] = functions
    return _bound[id(lib)]


class _CudaArray:
    """__cuda_array_interface__ over raw device memory: lets torch read an
    allocation of the sleep-mode pool as bytes without owning it."""

    def __init__(self, ptr, nbytes):
        self.__cuda_array_interface__ = {'shape': (nbytes,), 'typestr': '|u1', 'data': (ptr, False),
                                         'version': 3, 'strides': None}


class CudaDevice:
    def __init__(self, torch, device):
        from vllm.distributed.device_communicators.cuda_wrapper import CudaRTLibrary
        self.torch, self.device = torch, torch.device(device)
        self.index = self.device.index if self.device.index is not None else torch.cuda.current_device()
        self.rt = CudaRTLibrary()

    def bind(self):
        self.rt.cudaSetDevice(self.index)

    def register(self, ptr, nbytes):
        if self.rt.cudaHostRegister(ptr, nbytes, 0) != 0:
            self.rt.cudaGetLastError()
            return False
        return True

    def unregister(self, ptr):
        self.rt.cudaHostUnregister(ptr)

    def to_host(self, host_ptr, dev_ptr, nbytes):
        self.rt.cudaMemcpy(ctypes.c_void_p(host_ptr), ctypes.c_void_p(dev_ptr), nbytes)

    def to_device(self, dev_ptr, host_ptr, nbytes):
        self.rt.cudaMemcpy(ctypes.c_void_p(dev_ptr), ctypes.c_void_p(host_ptr), nbytes)

    # The pipelined reload's calls, bound from the same libcudart: vLLM's
    # wrapper has no streams or events.
    def _call(self, name, *args):
        fn = _runtime(self.rt.lib).get(name)
        err = fn(*args)
        if err != 0:
            message = self.rt.cudaGetErrorString(err)
            raise RuntimeError('%s: %s (%d)' % (name, message.decode() if isinstance(message, bytes) else message,
                                                err))

    def stream(self):
        handle = ctypes.c_void_p()
        self._call('cudaStreamCreateWithFlags', ctypes.byref(handle), _STREAM_NON_BLOCKING)
        return ('stream', handle)

    def event(self):
        handle = ctypes.c_void_p()
        self._call('cudaEventCreateWithFlags', ctypes.byref(handle), _EVENT_DISABLE_TIMING)
        return ('event', handle)

    def to_device_async(self, dev_ptr, host_ptr, nbytes, stream):
        self._call('cudaMemcpyAsync', ctypes.c_void_p(dev_ptr), ctypes.c_void_p(host_ptr), ctypes.c_size_t(nbytes),
                   _MEMCPY_HOST_TO_DEVICE, stream[1])

    def record(self, event, stream):
        self._call('cudaEventRecord', event[1], stream[1])

    def wait(self, event):
        self._call('cudaEventSynchronize', event[1])

    def drain(self, stream):
        self._call('cudaStreamSynchronize', stream[1])

    def destroy(self, handle):
        kind, value = handle
        self._call('cudaStreamDestroy' if kind == 'stream' else 'cudaEventDestroy', value)

    def synchronize(self):
        self.torch.cuda.synchronize(self.device)

    def sums(self, regions):
        """Fingerprints of (ptr, nbytes, stream position) regions, computed on
        the GPU in blocks; equal to pool.reference_sum."""
        torch = self.torch
        self.synchronize()
        out = []
        with torch.cuda.device(self.device), torch.inference_mode():
            for ptr, nbytes, position in regions:
                totals = torch.zeros(3, dtype=torch.int64, device=self.device)
                if nbytes:
                    view = torch.as_tensor(_CudaArray(ptr, nbytes), device=self.device)
                    for start in range(0, nbytes, BLOCK_BYTES):
                        n = min(BLOCK_BYTES, nbytes - start)
                        block = view.narrow(0, start, n).to(torch.int64)
                        weight = torch.arange(position + start, position + start + n, device=self.device,
                                              dtype=torch.int64).remainder_(251).add_(1)
                        totals[0] += block.sum()
                        totals[1] += block.square().sum()
                        totals[2] += block.mul_(weight).sum()
                        del block, weight
                out.append(totals)
            if not out:
                return []
            result = [[int(v) for v in row] for row in torch.stack(out).cpu().tolist()]
        # Hand the temporaries back: a restore wakes the KV cache next and
        # needs that memory free.
        del out
        torch.cuda.empty_cache()
        return result
