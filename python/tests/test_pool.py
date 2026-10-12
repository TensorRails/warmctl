import random
import unittest

from warmctl_vllm import pool


class Data:
    def __init__(self, size, tag, asleep=False):
        self.handle = (object(), size)
        self.tag = tag
        self.is_asleep = asleep


class PoolTest(unittest.TestCase):
    def table(self):
        return {0x7f00_0000_0000: Data(100, 'weights'), 0x7f00_1000_0000: Data(50, 'kv_cache'),
                0x7e00_0000_0000: Data(30, 'weights'), 0x7f00_2000_0000: Data(7, 'runtime')}

    def test_rows_sorted_by_address(self):
        rows = pool.allocation_rows(self.table())
        self.assertEqual([r['ptr'] for r in rows],
                         ['0x7e0000000000', '0x7f0000000000', '0x7f0010000000', '0x7f0020000000'])
        self.assertEqual(pool.layout(rows), [('0x7e0000000000', 30, 'weights'), ('0x7f0000000000', 100, 'weights')])

    def test_plan_and_pieces(self):
        rows = pool.allocation_rows(self.table())
        allocations, parts = pool.plan_parts(rows, part_bytes=64)
        self.assertEqual([(a['index'], a['bytes'], a['part'], a['offset']) for a in allocations],
                         [(0, 30, 0, 0), (1, 100, 0, 30)])
        self.assertEqual(parts, [{'index': 0, 'size': 64}, {'index': 1, 'size': 64}, {'index': 2, 'size': 2}])
        layout = pool.pieces(allocations, parts)
        self.assertEqual(layout, [[(0, 0, 0, 30), (1, 0, 30, 34)], [(1, 34, 0, 64)], [(1, 98, 0, 2)]])
        # A manifest that dropped the first allocation keeps the second in place.
        self.assertEqual(pool.pieces(allocations[1:], parts), [[(0, 0, 30, 34)], [(0, 34, 0, 64)], [(0, 98, 0, 2)]])
        self.assertEqual(pool.plan_parts([], 64), ([], []))
        with self.assertRaises(ValueError):
            pool.plan_parts(rows, part_bytes=pool.MAX_PART_BYTES + 1)

    def test_sums_are_additive_and_positional(self):
        data = bytes(random.Random(7).getrandbits(8) for _ in range(5000))
        whole = pool.reference_sum(data, 100)
        split = pool.total([pool.reference_sum(data[:1234], 100), pool.reference_sum(data[1234:], 1334)])
        self.assertEqual(whole, split)
        self.assertNotEqual(pool.reference_sum(data, 101), whole)
        changed = bytearray(data)
        changed[4321] ^= 1
        self.assertNotEqual(pool.reference_sum(bytes(changed), 100), whole)
        self.assertEqual(pool.compare([[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 7]]), [1])
        with self.assertRaises(ValueError):
            pool.compare([[1, 2, 3]], [])


if __name__ == '__main__':
    unittest.main()
