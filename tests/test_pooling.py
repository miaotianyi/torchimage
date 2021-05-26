import unittest
from typing import List

from numpy import ndarray

from torchimage.pooling.base import SeparablePoolNd
from torchimage.linalg import outer
from torchimage.filtering.base import SeparableFilterNd
from torchimage.padding import GenericPadNd

import numpy as np
import torch
from functools import reduce
from scipy import ndimage
from skimage import filters

NDIMAGE_PAD_MODES = [("symmetric", "reflect"),
                     ("replicate", "nearest"),
                     ("constant", "constant"),
                     ("reflect", "mirror"),
                     ("circular", "wrap")]


class MyTestCase(unittest.TestCase):
    def test_uniform(self):
        for n in range(1, 10):
            x = torch.rand(100, 41) * 100 - 50
            x = torch.round(x)
            for ti_mode, ndimage_mode in NDIMAGE_PAD_MODES:
                y_ti = SeparableFilterNd(np.ones(n) / n)(x, same=True, padder=GenericPadNd(mode=ti_mode))
                y_ndimage = ndimage.uniform_filter(x.numpy(), size=n, mode=ndimage_mode)
                result = np.allclose(y_ti.numpy(), y_ndimage, rtol=1e-5, atol=1e-5, equal_nan=False)
                with self.subTest(ti_mode=ti_mode, ndimage_mode=ndimage_mode, n=n):
                    self.assertTrue(result)

    def test_conv(self):
        for ti_mode, ndimage_mode in NDIMAGE_PAD_MODES:
            for ndim in range(1, 5):
                kernel_size = np.random.randint(1, 10, size=ndim)
                kernels = [np.random.rand(ks) for ks in kernel_size]
                shape = tuple(np.random.randint(20, 50, size=ndim))
                x = torch.rand(*shape)
                full_conv_tensor = reduce(outer, kernels)
                # note that convolve in neural network is correlate in signal processing
                y_ndimage = ndimage.correlate(x.numpy(), weights=full_conv_tensor, mode=ndimage_mode)
                y_ti = SeparableFilterNd(kernels)(x, same=True, padder=GenericPadNd(mode=ti_mode))
                result = np.allclose(y_ti.numpy(), y_ndimage, rtol=1e-5, atol=1e-5, equal_nan=False)
                with self.subTest(ti_mode=ti_mode, ndimage_mode=ndimage_mode, ndim=ndim, kernel_size=kernel_size, shape=shape):
                    self.assertTrue(result)


if __name__ == '__main__':
    unittest.main()
