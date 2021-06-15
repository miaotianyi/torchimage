import unittest

import numpy as np
import torch

from torchimage.padding import Padder
from torchimage.pooling import AveragePoolNd
from torchimage.shapes.conv_like import n_original_elements_1d


class MyTestCase(unittest.TestCase):
    @staticmethod
    def n_orignal_elements_gt(in_size, pad_width, kernel_size, stride):
        x = torch.ones(in_size, dtype=torch.int32)
        padder = Padder(pad_width=pad_width, mode="constant", constant_values=0)
        x = padder.forward(x, axes=None)
        print(x.tolist())
        return x.unfold(0, size=kernel_size, step=stride).sum(dim=-1).tolist()

    def test_n_original_elements_1d_0(self):
        expected = self.n_orignal_elements_gt(in_size=10, pad_width=(9, 6), kernel_size=4, stride=2)
        print(expected)

    def test_n_original_elements_1d(self):
        in_size = np.random.randint(1, 7)
        pad_width = np.random.randint(0, 7, size=2).tolist()
        kernel_size = np.random.randint(1, 7)
        stride = np.random.randint(1, 4)

        # in_size, pad_width, kernel_size, stride = 19, [0, 8], 3, 1
        # in_size=3; pad_width=[1, 6]; kernel_size=6; stride=3
        # in_size = 6; pad_width = [3, 9]; kernel_size = 2; stride = 1
        # in_size = 2; pad_width = [1, 1]; kernel_size = 5; stride=1
        # in_size = 1; pad_width = [4, 6]; kernel_size = 1; stride = 3
        # in_size=1; pad_width=[5, 2]; kernel_size=2; stride=3
        # in_size=3; pad_width=[1, 4]; kernel_size=6; stride=1

        if sum(pad_width) + in_size < kernel_size:
            return

        with self.subTest(in_size=in_size, pad_width=pad_width, kernel_size=kernel_size, stride=stride):
            print(f"{in_size=}; {pad_width=}; {kernel_size=}; {stride=}")
            expected = self.n_orignal_elements_gt(in_size=in_size, pad_width=pad_width, kernel_size=kernel_size, stride=stride)
            print(f"{expected=}")
            actual = n_original_elements_1d(in_size=in_size, pad_width=pad_width, kernel_size=kernel_size, stride=stride)
            print(f"{actual=}")
            self.assertEqual(expected, [a if a != '' else e for a, e in zip(actual, expected)])

    def test_n_original_elements_1d_repeated(self):
        for i in range(100):
            self.test_n_original_elements_1d()

    def test_n_original_elements_nd(self):
        # average pooling, such that border cases (for instance) has smaller re-normalization weight
        ndim = np.random.randint(1, 6)
        shape = np.random.randint(10, 30, size=ndim).tolist()
        kernel_size = np.random.randint(2, 8, size=ndim)
        stride = np.random.randint(2, 8, size=ndim)

        old_layer = AveragePoolNd(kernel_size, stride=stride, same_padder=Padder(mode="constant", constant_values=0))

        expected = old_layer.forward(torch.ones(tuple(shape)), axes=None)



if __name__ == '__main__':
    unittest.main()
