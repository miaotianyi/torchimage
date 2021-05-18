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
                self.assertEqual(len(nds), 4)
                self.assertEqual(len([x for x in nds]), 4)
                self.assertFalse(nds.is_item)
                self.assertEqual(len(nds), len(filter_list))
                for i in range(-len(filter_list), len(filter_list)):
                    # these objects shouldn't change at all
                    self.assertIs(filter_list[i], nds[i])

    def test_empty(self):
        with self.assertRaises(ValueError):
            NdSpec([])  # cannot accept empty input

    def test_list_singleton(self):
        nds = NdSpec([[1, 2]], item_shape=[2])
        self.assertFalse(nds.is_item)
        self.assertEqual(len(nds), 1)
        self.assertEqual(nds[0], [1, 2])
        self.assertEqual(nds[-1], [1, 2])
        self.assertEqual([x for x in nds], [[1, 2]])
        for i in list(range(-20, -1)) + list(range(1, 15)):
            with self.assertRaises(IndexError):
                nds[i]

    def test_item_list(self):
        nds = NdSpec([1, 2], item_shape=[2])
        self.assertEqual(len(nds), 0)
        self.assertEqual([x for x in nds], [[1, 2]])
        self.assertTrue(nds.is_item)
        for i in np.random.randint(-100, 100, 50):
            self.assertEqual(nds[i], [1, 2])

    def test_item_scalar(self):
        nds = NdSpec(423, item_shape=[])
        self.assertEqual(len(nds), 0)
        self.assertEqual([x for x in nds], [423])
        self.assertTrue(nds.is_item)
        for i in np.random.randint(-100, 100, 50):
            self.assertEqual(nds[i], 423)

    def test_broadcast_1(self):
        nds = NdSpec(423, item_shape=[2])
        self.assertTrue(nds.is_item)
        self.assertEqual(len(nds), 0)
        self.assertEqual([x for x in nds], [(423, 423)])
        for i in range(-10, 10):
            self.assertEqual(nds[-i], (423, 423))

    def test_broadcast_2(self):
        nds = NdSpec([1, 2, 4], item_shape=[2, 3])
        self.assertTrue(nds.is_item)
        self.assertEqual(len(nds), 0)
        self.assertEqual(list(nds), [([1, 2, 4], [1, 2, 4])])
        for i in range(-10, 10):
            self.assertEqual(nds[-i], ([1, 2, 4], [1, 2, 4]))

    def test_broadcast_3(self):
        nds = NdSpec("hello", item_shape=[1, 2, 3])
        self.assertTrue(nds.is_item)
        self.assertEqual(len(nds), 0)
        self.assertEqual([((("hello",) * 3,) * 2,)], list(nds))
        for i in range(-10, 10):
            self.assertEqual(nds[i], ((("hello",) * 3,) * 2,))

    def test_filter_list(self):
        data_1 = [[1, 2], [3, 4, 5], [6, 7, 8, 9]]
        data_2 = [[1, 2], (3, 4), np.array([5, 6]), torch.tensor([7., 8.], requires_grad=True)]
        for expected, actual in zip(data_1, NdSpec(data_1, item_shape=[-1])):
            self.assertEqual(expected, actual)

        for expected, actual in zip(data_2, NdSpec(data_2, item_shape=[-1])):
            self.assertIs(expected, actual)

    def test_filter_list_2(self):
        a = NdSpec(2, item_shape=[-1])
        self.assertEqual((2,), a[0])
        self.assertEqual("NdSpec(data=(2,), item_shape=(-1,))", str(a))


if __name__ == '__main__':
    unittest.main()
