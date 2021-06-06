import unittest

import torch
import numpy as np

from skimage import filters
from scipy import ndimage

from torchimage.filtering import edges
from torchimage.padding import Padder


NDIMAGE_PAD_MODES = [("symmetric", "reflect"),
                     ("replicate", "nearest"),
                     ("constant", "constant"),
                     ("reflect", "mirror"),
                     ("circular", "wrap")]


class MyTestCase(unittest.TestCase):
    def test_edge_filters(self):
        for ti_cls, ski_h, ski_v, ski_mag in [
            (edges.Sobel, filters.sobel_h, filters.sobel_v, filters.sobel),
            (edges.Prewitt, filters.prewitt_h, filters.prewitt_v, filters.prewitt),
            (edges.Scharr, filters.scharr_h, filters.scharr_v, filters.scharr),
            (edges.Farid, filters.farid_h, filters.farid_v, filters.farid),
        ]:
            x1 = torch.rand(np.random.randint(10, 50), np.random.randint(10, 50), dtype=torch.float64)
            x2 = x1.numpy()

            ti_filter = ti_cls(normalize=True, padder=Padder(mode="symmetric"))

            # in skimage, vertical edges go from the top to the bottom
            # therefore, it represents a sharp HORIZONTAL change in color
            # so the edge axis should be the horizontal (axis=-1)
            actual_h = ti_filter.horizontal(x1).numpy()
            expected_h = ski_h(x2)
            with self.subTest(shape=x1.shape, detector=ti_cls, task="horizontal"):
                self.assertLess(np.abs(actual_h - expected_h).max(), 1e-12)
            actual_v = ti_filter.vertical(x1).numpy()
            expected_v = ski_v(x2)
            with self.subTest(shape=x1.shape, detector=ti_cls, task="vertical"):
                self.assertLess(np.abs(actual_v - expected_v).max(), 1e-12)

            actual_mag = ti_filter.magnitude(x1, epsilon=0.0).numpy()
            expected_mag = ski_mag(x2) * np.sqrt(2)
            with self.subTest(shape=x1.shape, detector=ti_cls, task="magnitude"):
                self.assertLess(np.abs(actual_mag - expected_mag).max(), 1e-12)

    def test_ndimage_sobel(self):
        for ti_mode, ndimage_mode in NDIMAGE_PAD_MODES:
            x1 = torch.rand(np.random.randint(10, 50), np.random.randint(10, 50), dtype=torch.float64)
            x2 = x1.numpy()

            sobel = edges.Sobel(normalize=False, padder=Padder(mode=ti_mode))
            y_expected = ndimage.sobel(x2, axis=-1, mode=ndimage_mode)
            y_actual = sobel.component(x1, -1, -2).numpy()
            with self.subTest(shape=x1.shape, edge_axis=-1, mode=ti_mode, name="sobel"):
                self.assertLess(np.abs(y_expected - y_actual).max(), 1e-12)

            y_expected = ndimage.sobel(x2, axis=-2, mode=ndimage_mode)
            y_actual = sobel.component(x1, -2, -1).numpy()
            with self.subTest(shape=x1.shape, edge_axis=-2, mode=ti_mode, name="sobel"):
                self.assertLess(np.abs(y_expected - y_actual).max(), 1e-12)

    def test_ndimage_prewitt(self):
        for ti_mode, ndimage_mode in NDIMAGE_PAD_MODES:
            x1 = torch.rand(np.random.randint(10, 50), np.random.randint(10, 50), dtype=torch.float64)
            x2 = x1.numpy()

            prewitt = edges.Prewitt(normalize=False, padder=Padder(mode=ti_mode))
            y_expected = ndimage.prewitt(x2, axis=-1, mode=ndimage_mode)
            y_actual = prewitt.component(x1, -1, -2).numpy()
            with self.subTest(shape=x1.shape, edge_axis=-1, mode=ti_mode, name="sobel"):
                self.assertLess(np.abs(y_expected - y_actual).max(), 1e-12)

            y_expected = ndimage.prewitt(x2, axis=-2, mode=ndimage_mode)
            y_actual = prewitt.component(x1, -2, -1).numpy()
            with self.subTest(shape=x1.shape, edge_axis=-2, mode=ti_mode, name="sobel"):
                self.assertLess(np.abs(y_expected - y_actual).max(), 1e-12)

    def test_gaussian_gradient(self):
        sigma = 1
        truncate = 8
        for ti_mode, ndimage_mode in NDIMAGE_PAD_MODES:
            ndim = np.random.randint(2, 5)
            shape = np.random.randint(10, 20, size=ndim)
            x1 = torch.rand(*shape, dtype=torch.float64)
            x2 = x1.numpy()
            expected = ndimage.gaussian_gradient_magnitude(x2, sigma=sigma, mode=ndimage_mode, truncate=truncate)
            gg = edges.GaussianGrad(sigma=sigma, kernel_size=int(truncate*sigma*2+1), padder=Padder(mode=ti_mode))
            # epsilon = 0 for precision
            actual = gg.magnitude(x1, axes=None, epsilon=0.0).numpy()
            with self.subTest(mode=ti_mode, shape=shape, sigma=sigma, truncate=truncate):
                self.assertLess(np.abs(expected - actual).max(), 1e-15)

    def test_laplace_2(self):
        for ti_mode, ndimage_mode in NDIMAGE_PAD_MODES:
            laplace_2 = edges.Laplace(padder=Padder(mode=ti_mode))
            x1 = torch.rand(30, 23, dtype=torch.float64)
            x2 = x1.numpy()
            y_actual = laplace_2.forward(x1, axes=None).numpy()
            y_expected = ndimage.laplace(x2, mode=ndimage_mode)
            with self.subTest(mode=ti_mode):
                self.assertLess(np.abs(y_actual - y_expected).max(), 1e-15)

    def test_log(self):
        sigma = 1.5
        truncate = 2
        ks = int(sigma * truncate * 2 + 1)

        from matplotlib import pyplot as plt

        for ti_mode, ndimage_mode in NDIMAGE_PAD_MODES:
            x1 = torch.rand(17, 24, dtype=torch.float64)
            x2 = x1.numpy()
            y_actual = edges.LaplacianOfGaussian(kernel_size=ks, sigma=sigma, padder=Padder(mode=ti_mode)).forward(x1).numpy()
            y_expected = ndimage.gaussian_laplace(x2, sigma=sigma, mode=ndimage_mode, truncate=truncate)
            with self.subTest(mode=ti_mode):
                self.assertLess(np.abs(y_actual - y_expected).max(), 1e-15)


if __name__ == '__main__':
    unittest.main()
