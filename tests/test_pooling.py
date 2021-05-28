import unittest

from torchimage.linalg import outer
from torchimage.padding import GenericPadNd
from torchimage.pooling.base import SeparablePoolNd
from torchimage.pooling.gaussian import GaussianPoolNd
from torchimage.filtering.utils import _same_padding_pair
from torchimage.filtering.decorator import pool_to_filter


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


class SeparableFilterNd(SeparablePoolNd):
    # hand-written for testing only
    def __init__(self, kernel: object):
        super(SeparableFilterNd, self).__init__(kernel=kernel, stride=1)

    def forward(self, x: torch.Tensor, axes=None, same=True, padder: GenericPadNd = None):
        if same and padder is not None:
            # same padding
            same_pad_width = self.kernel_size.map(_same_padding_pair)
            padder = GenericPadNd(pad_width=same_pad_width,
                                  mode=padder.mode.data,
                                  constant_values=padder.constant_values.data,
                                  end_values=padder.end_values.data,
                                  stat_length=padder.stat_length.data)

        return super(SeparableFilterNd, self).forward(x, axes=axes, padder=padder)


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

    def test_wrapper_1(self):
        # wrapped image filter should behave the same way as its base pooling class
        GaussianFilterNd = pool_to_filter(GaussianPoolNd, same=True)

        x = torch.rand(17, 100, 5)

        # gaussian filter type
        gf_1 = GaussianFilterNd(9, sigma=1.5, order=0)
        gf_2 = GaussianFilterNd(9, 1.5, 0)
        gp = GaussianPoolNd(9, sigma=1.5, order=0, stride=1)

        y1 = gf_1(x, padder=GenericPadNd(mode="reflect"))
        y2 = gf_2(x, padder=GenericPadNd(mode="reflect"))
        y = gp(x, padder=GenericPadNd(pad_width=4, mode="reflect"))
        self.assertEqual(torch.abs(y1 - y).max().item(), 0)
        self.assertEqual(torch.abs(y2 - y).max().item(), 0)

    def test_gaussian_1(self):
        sigma = 1.5
        truncate = 4
        order = 0
        GaussianFilterClass = pool_to_filter(GaussianPoolNd, same=True)
        for ti_mode, ndimage_mode in NDIMAGE_PAD_MODES:
            print(int(truncate * sigma + 0.5))
            x = torch.rand(10, 37, 21, dtype=torch.float64)
            y_sp = ndimage.gaussian_filter(x.numpy(), sigma=sigma, order=order, mode=ndimage_mode, truncate=truncate)
            gf1 = GaussianFilterClass(kernel_size=2 * truncate * sigma + 1, sigma=sigma, order=order)
            print(f"{gf1.kernel_size=}")
            y_ti = gf1(x, axes=None, padder=GenericPadNd(mode=ti_mode))
            y_ti = y_ti.numpy()
            print(np.abs(y_sp - y_ti).max())

    def test_precision_1(self):
        # 1d convolution precision testing
        for ti_mode, ndimage_mode in NDIMAGE_PAD_MODES:
            x = torch.rand(10, dtype=torch.float64)
            w = torch.rand(5, dtype=torch.float64)
            y1 = ndimage.correlate1d(x.numpy(), w.numpy(), axis=-1, mode=ndimage_mode, origin=0)
            y2 = pool_to_filter(SeparablePoolNd, same=True)(w)(x, padder=GenericPadNd(mode=ti_mode)).numpy()
            result = np.allclose(y1, y2, rtol=1e-9, atol=1e-9)
            with self.subTest(ti_mode=ti_mode, ndimage_mode=ndimage_mode):
                self.assertTrue(result)





if __name__ == '__main__':
    unittest.main()
