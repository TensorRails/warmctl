"""What the visible GPUs support, read from the CUDA driver before an engine
starts: `python3 -m warmctl_vllm.devices` prints one JSON line. It calls
cuInit only, so it creates no context and takes no device memory."""
from __future__ import annotations

import ctypes
import json
import sys

# CUdevice_attribute values from cuda.h.
COMPUTE_MAJOR, COMPUTE_MINOR = 75, 76
VMM, POSIX_FD, MULTICAST = 102, 103, 132


class Driver:
    def __init__(self, lib=None):
        self.lib = lib if lib is not None else ctypes.CDLL('libcuda.so.1')

    def call(self, name, *args):
        result = getattr(self.lib, name)(*args)
        if result != 0:
            raise RuntimeError('%s failed with CUDA error %d' % (name, result))

    def count(self):
        n = ctypes.c_int()
        self.call('cuDeviceGetCount', ctypes.byref(n))
        return n.value

    def device(self, index):
        device = ctypes.c_int()
        self.call('cuDeviceGet', ctypes.byref(device), index)
        return device.value

    def attribute(self, device, attribute):
        value = ctypes.c_int()
        self.call('cuDeviceGetAttribute', ctypes.byref(value), attribute, device)
        return value.value

    def name(self, device):
        name = ctypes.create_string_buffer(256)
        self.call('cuDeviceGetName', name, len(name), device)
        return name.value.decode()


def describe(driver):
    """One row per visible GPU: what the interposer and fast collectives need."""
    driver.call('cuInit', 0)
    rows = []
    for index in range(driver.count()):
        device = driver.device(index)
        rows.append({
            'index': index,
            'name': driver.name(device),
            'compute_capability': '%d.%d' % (driver.attribute(device, COMPUTE_MAJOR),
                                             driver.attribute(device, COMPUTE_MINOR)),
            'vmm': bool(driver.attribute(device, VMM)),
            'posix_fd': bool(driver.attribute(device, POSIX_FD)),
            'multicast': bool(driver.attribute(device, MULTICAST)),
        })
    return rows


def main():
    sys.stdout.write(json.dumps({'devices': describe(Driver())}, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
