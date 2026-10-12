import os
import tempfile
import unittest

from warmctl_vllm import protocol


class ProtocolTest(unittest.TestCase):
    def test_roundtrip_sorted_compact(self):
        raw = protocol.encode({'b': [1, 2.5, 'x', None, True], 'a': {'k': 'v'}})
        self.assertEqual(raw, b'{"a":{"k":"v"},"b":[1,2.5,"x",null,true]}\n')
        self.assertEqual(protocol.decode(raw), {'a': {'k': 'v'}, 'b': [1, 2.5, 'x', None, True]})

    def test_refusals(self):
        with self.assertRaises(ValueError):
            protocol.encode({'x': float('nan')})
        with self.assertRaises(ValueError):
            protocol.encode({1: 'nonstring key'})
        with self.assertRaises(ValueError):
            protocol.decode(b'{"a":1,"a":2}')
        with self.assertRaises(ValueError):
            protocol.decode(b'{"a":NaN}')
        with self.assertRaises(ValueError):
            protocol.decode(b'')
        with self.assertRaises(ValueError):
            protocol.absolute('/a/../b')
        with self.assertRaises(ValueError):
            protocol.operation('../x')

    def test_publish_is_create_only(self):
        with tempfile.TemporaryDirectory() as root:
            name = os.path.join(root, 'status.json')
            record = protocol.publish(name, {'ok': True})
            self.assertEqual(record['bytes'], len(b'{"ok":true}\n'))
            with open(name, 'rb') as stream:
                self.assertEqual(stream.read(), b'{"ok":true}\n')
            with self.assertRaises(FileExistsError):
                protocol.publish(name, {'ok': False})
            self.assertEqual(sorted(os.listdir(root)), ['status.json'])


if __name__ == '__main__':
    unittest.main()
