import unittest

import numpy as np
import torch

from torchimage.padding import Padder
from torchimage.pooling import AveragePoolNd


class MyTestCase(unittest.TestCase):
    def test_n_original_elements_1d(self):
        pass

    def test_n_original_elements_nd(self):
        # average pooling, such that border cases (for instance) has smaller re-normalization weight
        ndim = np.random.randint(1, 6)
        shape = np.random.randint(10, 30, size=ndim)
        kernel_size = np.random.randint(2, 8, size=ndim)
        stride = np.random.randint(2, 8, size=ndim)

        old_layer = AveragePoolNd(kernel_size, stride=stride, same_padder=Padder(mode="constant", constant_values=0))

        expected = old_layer.forward(torch.ones(shape), axes=None)



if __name__ == '__main__':
    unittest.main()
