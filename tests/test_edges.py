import unittest

import torch
import numpy as np

from skimage import filters
from scipy import ndimage

from torchimage.filtering import edges
from torchimage.padding import GenericPadNd


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

            ti_filter = ti_cls(normalize=True)

            # in skimage, vertical edges go from the top to the bottom
            # therefore, it represents a sharp HORIZONTAL change in color
            # so the edge axis should be the horizontal (axis=-1)
            actual_h = ti_filter.horizontal(x1, same=True, padder=GenericPadNd(mode="symmetric")).numpy()
            expected_h = ski_h(x2)
            with self.subTest(shape=x1.shape, detector=ti_cls, task="horizontal"):
                self.assertLess(np.abs(actual_h - expected_h).max(), 1e-12)
            actual_v = ti_filter.vertical(x1, same=True, padder=GenericPadNd(mode="symmetric")).numpy()
            expected_v = ski_v(x2)
            with self.subTest(shape=x1.shape, detector=ti_cls, task="vertical"):
                self.assertLess(np.abs(actual_v - expected_v).max(), 1e-12)

            actual_mag = ti_filter.magnitude(x1, same=True, padder=GenericPadNd(mode="symmetric"), epsilon=0.0).numpy()
            expected_mag = ski_mag(x2) * np.sqrt(2)
            with self.subTest(shape=x1.shape, detector=ti_cls, task="magnitude"):
                self.assertLess(np.abs(actual_mag - expected_mag).max(), 1e-12)


if __name__ == '__main__':
    unittest.main()
