import unittest

from torchimage.pooling.base import SeparablePoolNd
from torchimage.filtering.base import SeparableFilterNd
from torchimage.padding import GenericPadNd

import numpy as np
import torch
from functools import reduce
from scipy import ndimage
from skimage import filters


class MyTestCase(unittest.TestCase):
    NDIMAGE_PAD_MODES = [("symmetric", "reflect"),
                         ("replicate", "nearest"),
                         ("constant", "constant"),
                         ("reflect", "mirror"),
                         ("circular", "wrap")]

    def test_uniform(self):
        cval = np.random.rand() * 1_000 - 5
        n = 3
        x = torch.rand(100, 41) * 100 - 50
        for ti_mode, ndimage_mode in self.NDIMAGE_PAD_MODES:
            y_ti = SeparableFilterNd(np.ones(n) / n)(x, same=True, padder=GenericPadNd(mode=ti_mode))
            y_ndimage = ndimage.uniform_filter(x.numpy(), size=n, mode=ndimage_mode)
            self.assertTrue(np.abs(y_ti.numpy() - y_ndimage).mean() < 1e-5)


if __name__ == '__main__':
    unittest.main()
