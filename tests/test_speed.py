import unittest
from timeit import timeit

import numpy as np
import torch
from torch.nn import functional as F

from torchimage.padding import Padder
from torchimage.utils import NdSpec
from torchimage.utils.validation import check_axes
from torchimage.shapes.conv_like import n_original_elements_nd
from torchimage.pooling import GaussianPoolNd, SeparablePoolNd
from torchimage.pooling.uniform import AvgPoolNd
from .test_pooling import NDIMAGE_PAD_MODES


class AveragePoolNd(SeparablePoolNd):
    @staticmethod
    def _get_kernel(kernel_size):
        return NdSpec(kernel_size).map(lambda ks: (np.ones(ks) / ks).tolist())

    def __init__(self, kernel_size, stride=None, *, count_include_pad=True, same_padder=None):
        super().__init__(
            kernel=AveragePoolNd._get_kernel(kernel_size=kernel_size),
            stride=stride,
            same_padder=same_padder
        )
        self.count_include_pad = count_include_pad

    def forward(self, x: torch.Tensor, axes=slice(2, None)):
        if self.count_include_pad:
            return super().forward(x, axes=axes)

        else:
            input_shape = x.shape
            z_old = np.prod([self.kernel_size[i] for i in range(len(axes))])

            axes = check_axes(x, axes)
            idx = np.argsort(axes)  # axes must be sorted
            assert len(axes) == len(set(axes))  # axes must be unique
            z = n_original_elements_nd(
                in_size=[input_shape[axes[i]] for i in idx],
                pad_width=[self.same_padder.pad_width[i] for i in idx],
                kernel_size=[self.kernel_size[i] for i in idx],
                stride=[self.stride[i] for i in idx],
                device=x.device, dtype=x.dtype
            )
            new_shape = [1] * x.ndim
            for i in idx:
                new_shape[axes[i]] = z.shape[i]
            z = z.view(*new_shape)

            x = super().forward(x, axes=axes) * z_old / z
            return x


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.gp = GaussianPoolNd(kernel_size=7, sigma=1.5, stride=1)

    def test_separable_pad(self):
        data = torch.rand(32, 32, 32)
        padder = Padder(pad_width=20, mode="reflect")

        def separable_pad():
            self.gp.forward(data, axes=None, padder=padder)

        def normal_pad():
            x = padder.forward(data, axes=None)
            self.gp.forward(x, axes=None)

        result_1 = timeit("f()", globals={"f": separable_pad}, number=1000)
        print(f"{result_1=}")
        result_2 = timeit("f()", globals={"f": normal_pad}, number=1000)
        print(f"{result_2=}")

    def test_average_speed(self):
        from timeit import timeit

        for kernel_size in range(3, 15, 2):
            x = torch.rand(1, 1, 13, 42, 58, dtype=torch.float64)
            old_filter = AveragePoolNd(kernel_size=kernel_size).to_filter(padder=None)
            new_filter = AvgPoolNd(kernel_size=kernel_size).to_filter(padder=None)
            t_old = timeit("f(x)", globals={"f": old_filter.forward, "x": x}, number=1000)
            # print(f"{t_old=}")
            t_new = timeit("f(x)", globals={"f": new_filter.forward, "x": x}, number=1000)
            # print(f"{t_new=}")
            t_base = timeit(f"f(x, kernel_size={kernel_size}, stride=1)", globals={"f": F.avg_pool3d, "x": x}, number=1000)
            print(f"{t_old=}, {t_new=}, {t_base=}")
            # print(f"{kernel_size=}, {ti_mode=}, {(t_old-t_new)/t_old=}")



if __name__ == '__main__':
    unittest.main()
