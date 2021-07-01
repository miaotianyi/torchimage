import unittest

from torchimage.misc import outer
from torchimage.pooling.base import SeparablePoolNd
from torchimage.pooling.gaussian import GaussianPoolNd
from torchimage.pooling.uniform import AvgPoolNd

from torchimage.padding.utils import same_padding_width


import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from functools import reduce
from scipy import ndimage

NDIMAGE_PAD_MODES = [("symmetric", "reflect"),
                     ("replicate", "nearest"),
                     ("constant", "constant"),
                     ("reflect", "mirror"),
                     ("circular", "wrap")]


class MyTestCase(unittest.TestCase):
    def test_uniform_1d(self):
        # x = torch.arange(10, dtype=torch.float64)
        x = torch.rand(30, dtype=torch.float64)
        for n in range(1, 10):
            for ti_mode, ndimage_mode in NDIMAGE_PAD_MODES:
                filter_layer = SeparablePoolNd(np.ones(n) / n).to_filter(ti_mode)
                y_ti = filter_layer.forward(x, axes=None).numpy()
                y_ndimage = ndimage.uniform_filter(x.numpy(), size=n, mode=ndimage_mode)
                with self.subTest(n=n, ti_mode=ti_mode):
                    self.assertLess(np.abs(y_ti - y_ndimage).max(), 1e-14)

    def test_uniform(self):
        for n in range(1, 10):
            x = torch.rand(100, 41, dtype=torch.float64) * 100 - 50
            x = torch.round(x)
            for ti_mode, ndimage_mode in NDIMAGE_PAD_MODES:
                filter_layer = SeparablePoolNd(np.ones(n) / n).to_filter(ti_mode)
                y_ti = filter_layer.forward(x, axes=None).numpy()
                y_ndimage = ndimage.uniform_filter(x.numpy(), size=n, mode=ndimage_mode)

                result = np.allclose(y_ti, y_ndimage, rtol=1e-5, atol=1e-5, equal_nan=False)

                with self.subTest(ti_mode=ti_mode, ndimage_mode=ndimage_mode, n=n):
                    self.assertTrue(result)

    def test_conv(self):
        for ti_mode, ndimage_mode in NDIMAGE_PAD_MODES:
            for ndim in range(1, 5):
                kernel_size = np.random.randint(1, 10, size=ndim)
                kernels = [np.random.rand(ks) for ks in kernel_size]
                shape = tuple(np.random.randint(20, 50, size=ndim))
                x = torch.rand(*shape, dtype=torch.float64)
                full_conv_tensor = reduce(outer, kernels)
                # note that convolve in neural network is correlate in signal processing
                y_ndimage = ndimage.correlate(x.numpy(), weights=full_conv_tensor, mode=ndimage_mode)
                filter_layer = SeparablePoolNd(kernels).to_filter(padder=ti_mode)
                y_ti = filter_layer.forward(x, axes=None).numpy()
                result = np.allclose(y_ti, y_ndimage, rtol=1e-7, atol=1e-5, equal_nan=False)

                with self.subTest(ti_mode=ti_mode, ndimage_mode=ndimage_mode, ndim=ndim,
                                  kernel_size=kernel_size, shape=shape):
                    self.assertTrue(result)

    def test_wrapper_1(self):
        # wrapped image filter should behave the same way as its base pooling class
        x = torch.rand(17, 100, 5)

        # gaussian filter type
        gf_1 = GaussianPoolNd(9, sigma=1.5, order=0).to_filter("reflect")
        gf_2 = GaussianPoolNd(9, 1.5, 0).to_filter("reflect")
        gp = GaussianPoolNd(9, sigma=1.5, order=0, stride=1, same_padder="reflect")

        y1 = gf_1.forward(x, axes=None)
        y2 = gf_2.forward(x, axes=None)
        y = gp.forward(x, axes=None)
        self.assertEqual(torch.abs(y1 - y).max().item(), 0)
        self.assertEqual(torch.abs(y2 - y).max().item(), 0)

    def test_gaussian_1(self):
        sigma = 1.5

        for truncate in range(2, 10, 2):
            for order in range(6):
                for ti_mode, ndimage_mode in NDIMAGE_PAD_MODES:
                    x = torch.rand(10, 37, 21, dtype=torch.float64)
                    y_sp = ndimage.gaussian_filter(x.numpy(), sigma=sigma, order=order, mode=ndimage_mode,
                                                   truncate=truncate)
                    gf1 = GaussianPoolNd(kernel_size=int(2 * truncate * sigma + 1), sigma=sigma, order=order,
                                         ).to_filter(padder=ti_mode)
                    y_ti = gf1.forward(x, axes=None)
                    y_ti = y_ti.numpy()
                    self.assertLess(np.abs(y_sp - y_ti).max(), 1e-10)

    def test_precision_1(self):
        # 1d convolution precision testing
        for ti_mode, ndimage_mode in NDIMAGE_PAD_MODES:
            x = torch.rand(10, dtype=torch.float64)
            w = torch.rand(5, dtype=torch.float64)
            y1 = ndimage.correlate1d(x.numpy(), w.numpy(), axis=-1, mode=ndimage_mode, origin=0)
            pool_layer = SeparablePoolNd(w).to_filter(padder=ti_mode)
            y2 = pool_layer.forward(x, axes=None).numpy()
            result = np.allclose(y1, y2, rtol=1e-9, atol=1e-9)
            with self.subTest(ti_mode=ti_mode, ndimage_mode=ndimage_mode):
                self.assertTrue(result)

    def test_average_1(self):
        for kernel_size in range(3, 15, 2):
            x = torch.rand(13, 25, 18, dtype=torch.float64)
            for ti_mode, ndimage_mode in NDIMAGE_PAD_MODES:
                filter_layer = AvgPoolNd(kernel_size=kernel_size).to_filter(padder=ti_mode)
                y_ti = filter_layer.forward(x, axes=None).numpy()
                y_ndi = ndimage.uniform_filter(x.numpy(), size=kernel_size, mode=ndimage_mode)
                with self.subTest(kernel_size=kernel_size, ti_mode=ti_mode, ndimage_mode=ndimage_mode):
                    self.assertLess(np.abs(y_ti - y_ndi).max(), 1e-10)

    def test_average_2(self):
        for kernel_size in range(3, 15, 2):
            x = torch.rand(1, 1, 13, 18, dtype=torch.float64)
            ti_mode = "constant"

            filter_layer = AvgPoolNd(kernel_size=kernel_size, count_include_pad=True).to_filter(padder=ti_mode)
            y_ti = filter_layer.forward(x).squeeze().numpy()
            y_torch = F.avg_pool2d(x, kernel_size=kernel_size, stride=1,
                                   padding=kernel_size//2,  count_include_pad=True).squeeze().numpy()
            with self.subTest(kernel_size=kernel_size, ti_mode=ti_mode, count_include_pad=True):
                self.assertLess(np.abs(y_ti - y_torch).max(), 1e-10)

            filter_layer = AvgPoolNd(kernel_size=kernel_size, count_include_pad=False).to_filter(padder=ti_mode)
            y_ti = filter_layer.forward(x).squeeze().numpy()
            y_torch = F.avg_pool2d(x, kernel_size=kernel_size, stride=1,
                                   padding=kernel_size // 2, count_include_pad=False).squeeze().numpy()
            with self.subTest(kernel_size=kernel_size, ti_mode=ti_mode, count_include_pad=False):
                self.assertLess(np.abs(y_ti - y_torch).max(), 1e-10)


if __name__ == '__main__':
    unittest.main()
