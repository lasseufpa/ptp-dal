import copy
import unittest

from ptp.compression import *

original_ds = {
    'metadata': {},
    'data': [{
        "t1": 0,
        "t2": 18,
        "t3": 30,
        "t4": 38
    }, {
        "t1": 1,
        "t2": 12,
        "t3": 31,
        "t4": 42,
        "v": 10,
        "v2": 11
    }, {
        "t1": 2,
        "t2": 16,
        "t3": 32,
        "t4": 45
    }, {
        "t1": 3,
        "t2": 19,
        "t3": 33,
        "t4": 55,
        "v": 10,
        "v2": 12
    }, {
        "t1": 4,
        "t2": 12,
        "t3": 34,
        "t4": 41
    }]
}
compressed_ds = {
    'metadata': {},
    'non-indexed': {
        't1': [0, 1, 2, 3, 4],
        't2': [18, 12, 16, 19, 12],
        't3': [30, 31, 32, 33, 34],
        't4': [38, 42, 45, 55, 41]
    },
    'indexed': {
        'v': [10, 10],
        'v2': [11, 12]
    },
    'idx': {
        'v': [1, 3],
        'v2': 'v'
    }
}


class TestCompression(unittest.TestCase):
    def test_compression(self):
        codec = Codec(copy.deepcopy(original_ds))
        mutable_ds = codec.compress()
        self.assertEqual(mutable_ds, compressed_ds)

    def test_decompression(self):
        codec = Codec(copy.deepcopy(compressed_ds), compressed=True)
        mutable_ds = codec.decompress()
        self.assertEqual(mutable_ds, original_ds)
