import unittest

import numpy as np
import torch

from torchimage.padding import Padder
from torchimage.pooling import AveragePoolNd
from torchimage.shapes.conv_like import n_original_elements_1d


class MyTestCase(unittest.TestCase):
    def test_n_original_elements_1d(self):
        in_size = np.random.randint(1, 20)
        pad_before, pad_after = np.random.randint(0, 10, size=2)
        kernel_size = np.random.randint(1, 7)
        stride = np.random.randint(1, 4)

        with self.subTest(in_size=in_size, pad_before=pad_before, pad_after=pad_after, kernel_size=kernel_size, stride=stride):
            # set up expected
            x = torch.ones(in_size)
            padder = Padder(pad_width=(pad_before, pad_after), mode="constant", constant_values=0)
            x = padder.forward(x, axes=None)

            expected = x.unfold(0, size=kernel_size, step=stride).sum(dim=-1)
            actual = n_original_elements_1d(in_size=in_size, pad_width=(pad_before, pad_after), kernel_size=kernel_size, stride=stride)
            print(expected)
            print(actual)

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
