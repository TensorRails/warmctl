import unittest

from warmctl_vllm import devices


class FakeDriver(devices.Driver):
    """Two GPUs: an H100 (VMM, POSIX handles, multicast) and a GPU without VMM."""

    attributes = {
        (0, devices.COMPUTE_MAJOR): 9, (0, devices.COMPUTE_MINOR): 0, (0, devices.VMM): 1,
        (0, devices.POSIX_FD): 1, (0, devices.MULTICAST): 1,
        (1, devices.COMPUTE_MAJOR): 8, (1, devices.COMPUTE_MINOR): 9, (1, devices.VMM): 0,
        (1, devices.POSIX_FD): 0, (1, devices.MULTICAST): 0,
    }

    def __init__(self):
        self.calls = []

    def call(self, name, *args):
        self.calls.append(name)

    def count(self):
        return 2

    def device(self, index):
        return index

    def attribute(self, device, attribute):
        return self.attributes[(device, attribute)]

    def name(self, device):
        return ['NVIDIA H100 80GB HBM3', 'vGPU'][device]


class DevicesTest(unittest.TestCase):
    def test_describe(self):
        driver = FakeDriver()
        rows = devices.describe(driver)
        self.assertEqual(driver.calls, ['cuInit'])
        self.assertEqual(rows[0], {'index': 0, 'name': 'NVIDIA H100 80GB HBM3', 'compute_capability': '9.0',
                                   'vmm': True, 'posix_fd': True, 'multicast': True})
        self.assertEqual((rows[1]['compute_capability'], rows[1]['vmm'], rows[1]['posix_fd']), ('8.9', False, False))

    def test_errors_name_the_call(self):
        class Lib:
            @staticmethod
            def cuInit(flags):
                return 100

        with self.assertRaisesRegex(RuntimeError, 'cuInit failed with CUDA error 100'):
            devices.describe(devices.Driver(Lib()))


if __name__ == '__main__':
    unittest.main()
