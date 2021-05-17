import unittest

import numpy as np
import torch
from torchimage.utils.ragged import get_ragged_ndarray, expand_ragged_ndarray


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.n_trials = 20

    def test_get_shape_1(self):
        f = get_ragged_ndarray
        # test regular arrays
        for _ in range(self.n_trials):
            ndim = np.random.randint(0, 10)
            shape = np.random.randint(0, 5, ndim)
            with self.subTest(shape=shape):
                self.assertEqual(f(np.empty(shape))[1], tuple(shape))

    def test_get_shape_2(self):
        f = get_ragged_ndarray
        with self.assertRaises(ValueError):
            data, shape = f([1, [2, 3]], strict=True)

        original = [np.random.rand(np.random.randint(0, 8)) for _ in range(100)]
        data, shape = f(original, strict=False)
        self.assertEqual(shape, (100, -1))
        for a, b in zip(data, original):
            self.assertTrue(np.array_equal(a, b))

        data, shape = f((1, (2, 3)), strict=False)
        self.assertEqual(data, ((1,), (2, 3)))
        self.assertEqual(shape, (2, -1))

        data, shape = f((1, (2, (3, 4), 5)), strict=False)
        self.assertEqual(data, (((1,),), ((2, ), (3, 4), (5,))))
        self.assertEqual(shape, (2, -1, -1))

        data, shape = f(((1, 2), ((3, 4), (5, 6))), strict=False)
        self.assertEqual(data, (((1, 2),), ((3, 4), (5, 6))))
        self.assertEqual(shape, (2, -1, 2))

    def test_expand(self):
        for new_shape in [
            [1, 2, -1],
            [1, -1, -1]
        ]:
            arr, shape = expand_ragged_ndarray([[1], [2, 3]], old_shape=[2, -1], new_shape=new_shape)
            self.assertEqual(arr, (((1,), (2, 3)),))
            self.assertEqual(shape, (1, 2, -1))

        arr, shape = expand_ragged_ndarray([[1], [2], [3]], old_shape=[-1, -1], new_shape=[1, 3, 6])
        self.assertEqual(arr, (((1,) * 6,) + ((2,) * 6,) + ((3,) * 6,),))
        self.assertEqual(shape, (1, 3, 6))

        arr, shape = get_ragged_ndarray("hello")
        self.assertEqual(arr, "hello")
        self.assertEqual(shape, ())

        arr, shape = expand_ragged_ndarray("hello", old_shape=(), new_shape=[1, 3, 6])
        self.assertEqual(arr, ((("hello",) * 6,) * 3,))
        self.assertEqual(shape, (1, 3, 6))

    def test_filter_list_1(self):
        data = [[1, 2], [3, 4, 5], [6, 7, 8, 9]]
        arr, shape = get_ragged_ndarray(data, strict=True)
        actual_arr, actual_shape = expand_ragged_ndarray(arr, shape, new_shape=[-1])
        self.assertEqual(actual_arr, tuple(tuple(x) for x in data))
        self.assertEqual(actual_shape, (3, -1))

    def test_filter_list_2(self):
        data = [[1, 2], [3, 4], [5, 6]]
        arr, shape = get_ragged_ndarray(data, strict=True)
        actual_arr, actual_shape = expand_ragged_ndarray(arr, shape, new_shape=[-1])
        self.assertEqual(actual_arr, tuple(tuple(x) for x in data))
        self.assertEqual(actual_shape, (3, 2))


if __name__ == '__main__':
    unittest.main()

