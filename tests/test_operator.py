import unittest

import numpy as np
import torch

from torchimage.pooling.operator import any_conv_1d


class MyTestCase(unittest.TestCase):
    @staticmethod
    def gt_any_conv_1d(x, w, dim: int, stride: int):
        return x.unfold(dim, size=len(w), step=stride) @ w

    def test_something(self):
        for i in range(10):
            ndim = np.random.randint(1, 6)
            shape = np.random.randint(1, 20, size=ndim)
            x = torch.rand(*shape, dtype=torch.float64)

            dim = np.random.randint(0, ndim)
            stride = np.random.randint(1, 5)

            kernel_size = np.random.randint(1, shape[dim] + 1)
            w = torch.rand(kernel_size, dtype=torch.float64)
            expected = self.gt_any_conv_1d(x, w, dim=dim, stride=stride)
            actual = any_conv_1d(x, w, dim=dim, stride=stride, dilation=1)
            with self.subTest(i=i):
                self.assertTrue(torch.allclose(expected, actual))


if __name__ == '__main__':
    unittest.main()
