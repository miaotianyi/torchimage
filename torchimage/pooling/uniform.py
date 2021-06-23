import numpy as np
import torch

from .base import SeparablePoolNd

from ..utils import NdSpec
from ..utils.validation import check_axes
from ..shapes.conv_like import n_original_elements_nd


class AvgPoolNd(SeparablePoolNd):
    @staticmethod
    def _get_kernel(kernel_size):
        return NdSpec(kernel_size).map(lambda ks: (np.ones(ks) / ks).tolist())

    def __init__(self, kernel_size, stride=None, *, count_include_pad=True, same_padder=None):
        super().__init__(
            kernel=self._get_kernel(kernel_size=kernel_size),
            stride=stride,
            same_padder=same_padder
        )
        self.count_include_pad = count_include_pad

    def forward(self, x: torch.Tensor, axes=slice(2, None)):
        if self.count_include_pad:
            return super().forward(x, axes=axes)

        else:
            input_shape = x.shape
            axes = check_axes(x, axes)

            z_old = np.prod([self.kernel_size[i] for i in range(len(axes))])

            idx = np.argsort(axes)  # axes must be sorted
            assert len(axes) == len(set(axes))  # axes must be unique

            x = super().forward(x, axes=axes)
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

            x = x * z_old / z
            return x
