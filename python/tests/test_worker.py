import json
import os
import tempfile
import unittest

from warmctl_vllm.worker import Extension


class Rank(Extension):
    def __init__(self, rank):
        self.rank = rank

    def _warmctl_rank(self):
        return self.rank

    def checkpoint_prepare(self):
        return None

    def echo(self, value):
        return {'rank': self.rank, 'value': value}


class ReplyTest(unittest.TestCase):
    def test_every_rank_publishes_its_result(self):
        with tempfile.TemporaryDirectory() as root:
            reply = os.path.join(root, 'rpc', 'echo-1')
            for rank in range(3):
                self.assertEqual(Rank(rank).warmctl_rpc('echo', {'value': 7}, reply), {'rank': rank, 'value': 7})
            self.assertEqual(Rank(3).warmctl_rpc('checkpoint_prepare', {}, reply), None)
            self.assertEqual(sorted(os.listdir(reply)), ['0.json', '1.json', '2.json', '3.json'])
            self.assertEqual(os.stat(os.path.join(reply, '0.json')).st_mode & 0o777, 0o644)
            with open(os.path.join(reply, '2.json')) as stream:
                self.assertEqual(json.load(stream), {'result': {'rank': 2, 'value': 7}})
            with open(os.path.join(reply, '3.json')) as stream:
                self.assertEqual(json.load(stream), {'result': None})

    def test_refusals(self):
        with tempfile.TemporaryDirectory() as reply:
            Rank(0).warmctl_rpc('echo', {'value': 1}, reply)
            with self.assertRaises(FileExistsError):
                Rank(0).warmctl_rpc('echo', {'value': 2}, reply)
            with self.assertRaises(ValueError):
                Rank(1).warmctl_rpc('_warmctl_rank', {}, reply)
            with self.assertRaises(ValueError):
                Rank(1).warmctl_rpc('echo', {'value': 1}, 'relative/dir')
            self.assertEqual(os.listdir(reply), ['0.json'])


if __name__ == '__main__':
    unittest.main()


class ReplicaTest(unittest.TestCase):
    def test_dense_replicas_use_the_index(self):
        from types import SimpleNamespace

        from warmctl_vllm.worker import replica
        # A dense model on vLLM with data_parallel_index: every replica has
        # data_parallel_rank 0.
        self.assertEqual(replica(SimpleNamespace(data_parallel_rank=0, data_parallel_index=5)), 5)
        # Expert parallelism keeps both equal.
        self.assertEqual(replica(SimpleNamespace(data_parallel_rank=3, data_parallel_index=3)), 3)
        # Older vLLM without the field.
        self.assertEqual(replica(SimpleNamespace(data_parallel_rank=2)), 2)
