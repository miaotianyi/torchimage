import unittest
import numpy as np
from torchimage.utils import NdSpec
import torch


class MyTestCase(unittest.TestCase):
    def test_kernels(self):
        nds = NdSpec([1, 2, 3], item_shape=(-1, ))
        self.assertTrue(nds.is_item)
        for i in range(-10, 10):
            self.assertEqual(nds[i], [1, 2, 3])

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
        # each item is a list of unknown length; can be empty (is empty in this case)
        data = []
        a = NdSpec(data, item_shape=(-1,))
        self.assertIs(a[0], data)
        # with self.assertRaises(ValueError):
        with self.assertRaises(ValueError):
            NdSpec([], item_shape=(1, 2, 3))  # cannot accept empty input

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
        self.assertEqual("NdSpec(data=(2,), item_shape=(-1,), index_shape=())", str(a))

    def test_index_tuple_1(self):
        d1 = [1, 2]
        d2 = [3, 4, 5]
        a = NdSpec([d1, d2], item_shape=())
        self.assertIs(a[0], d1)
        self.assertIs(a[1], d2)
        for i, val in enumerate(d1):
            self.assertEqual(a[0, i], val)
        for i, val in enumerate(d2):
            self.assertEqual(a[1, i], val)

    def test_map_1(self):
        for _ in range(10):
            a = np.random.rand(10)
            nds = NdSpec(a, item_shape=[])
            self.assertTrue(np.array_equal(a * 2, nds.map(lambda x: x * 2)))

    def test_zip_1(self):
        a1 = NdSpec([1, 2, 3])
        a2 = NdSpec([4, 5, 6], item_shape=[-1])
        a3 = NdSpec([[7, 8], [9, 10], [11]], item_shape=[-1])
        actual = NdSpec.zip(a1, a2, a3)
        expected = NdSpec(data=[(1, [4, 5, 6], [7, 8]),
                                (2, [4, 5, 6], [9, 10]),
                                (3, [4, 5, 6], [11])], item_shape=(3,))

        self.assertEqual(expected.data, actual.data)
        self.assertEqual(expected.item_shape, actual.item_shape)

    def test_zip_2(self):
        a1 = NdSpec([1, 2, 3])
        a2 = NdSpec([4, 5, 6, 7])
        with self.assertRaises(ValueError):
            NdSpec.zip(a1, a2)


if __name__ == '__main__':
    unittest.main()
