import unittest
import numpy as np
from torchimage.utils import NdSpec
import torch


class MyTestCase(unittest.TestCase):
    def test_ragged(self):
        for filter_list in [
            [[1, 2, 3], [4, 5], [6, 7, 8], [9]],
            ([1, 2, 3], [4, 5], [6, 7, 8], [9]),
            ([1, 2, 3], (4, 5), [6, 7, 8], (9,)),
            ([1, 2, 3], (4, 5), [6, 7, 8], 9),
            (np.array([1, 2, 3]), (4, 5), [6, 7, 8], 9),
            (torch.tensor([1, 2, 3]), (4, 5), [6, 7, 8], 9),
            (torch.tensor([[1, 2], [2, 3]]), (4, 5), [6, 7, 8], 9),
        ]:
            with self.subTest(data=filter_list):
                nds = NdSpec(filter_list, item_shape=())
                print(nds)
                self.assertFalse(nds.is_item)
                self.assertEqual(len(nds), len(filter_list))
                for i in range(-len(filter_list), len(filter_list)):
                    self.assertTrue(np.array_equal(nds(i), filter_list[i]))

    def test_empty(self):
        with self.assertRaises(ValueError):
            NdSpec([])  # cannot accept empty input

    def test_list_singleton(self):
        nds = NdSpec([[1, 2]], item_shape=[2])
        print(nds)
        self.assertFalse(nds.is_item)
        self.assertEqual(nds(0), [1, 2])
        self.assertEqual(nds(-1), [1, 2])
        for i in list(range(-20, -1)) + list(range(1, 15)):
            with self.assertRaises(IndexError):
                nds(i)

    def test_item_list(self):
        nds = NdSpec([1, 2], item_shape=[2])
        print(nds)
        self.assertTrue(nds.is_item)
        for i in np.random.randint(-100, 100, 50):
            self.assertEqual(nds(i), [1, 2])

    def test_item_scalar(self):
        nds = NdSpec(423, item_shape=[])
        print(nds)
        self.assertTrue(nds.is_item)
        for i in np.random.randint(-100, 100, 50):
            self.assertEqual(nds(i), 423)

    def test_broadcast_1(self):
        nds = NdSpec(423, item_shape=[2])
        print(nds)
        self.assertTrue(nds.is_item)
        for i in range(-10, 10):
            self.assertEqual(nds(-i), [423, 423])

    def test_broadcast_2(self):
        nds = NdSpec([1, 2, 4], item_shape=[2, 3])
        print(nds)
        self.assertTrue(nds.is_item)
        for i in range(-10, 10):
            self.assertEqual(nds(-i), [[1, 2, 4], [1, 2, 4]])

    def test_broadcast_3(self):
        nds = NdSpec("hello", item_shape=[1, 2, 3])
        print(nds)
        self.assertTrue(nds.is_item)
        for i in range(-10, 10):
            self.assertEqual(nds(i), [[["hello"] * 3] * 2])


if __name__ == '__main__':
    unittest.main()
