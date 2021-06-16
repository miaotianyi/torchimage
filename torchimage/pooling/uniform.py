import warnings

import numpy as np
import torch

from .base import BasePoolNd, SeparablePoolNd

from ..padding import Padder
from ..utils import NdSpec
from ..utils.validation import check_axes
from ..shapes.conv_like import n_original_elements_nd


class AveragePoolNd(SeparablePoolNd):
    # TODO: this average method doesn't have an option to count_include_pad
    #   For example, with kernel_size=3, we should allow the corner pixel
    #   to be the average of 4 instead of 9 nearby pixels

    @staticmethod
    def _get_kernel(kernel_size):
        return NdSpec(kernel_size).map(lambda ks: (np.ones(ks) / ks).tolist())

    def __init__(self, kernel_size, stride=None, *, same_padder=None):
        super().__init__(
            kernel=AveragePoolNd._get_kernel(kernel_size=kernel_size),
            stride=stride,
            same_padder=same_padder
        )


class AvgPoolNd(BasePoolNd):
    def __init__(self, kernel_size, stride=None, *, count_include_pad=True,
                 same_padder=Padder(mode="constant", constant_values=0)):
        super().__init__(same_padder=same_padder)
        self.kernel_size = NdSpec(kernel_size, item_shape=[])
        self.stride = self.read_stride(stride)

        self.count_include_pad = count_include_pad

        self._align_params()

    def _align_params(self):
        self.ndim = NdSpec.agg_index_len(self.kernel_size, self.stride, allowed_index_ndim=(0, 1))

    def forward(self, x: torch.Tensor, axes=slice(2, None)):
        axes = check_axes(x, axes)
        assert self.ndim == 0 or self.ndim == len(axes)

        input_shape = x.shape

        x = self.pad(x, axes=axes)

        if self.count_include_pad:
            z = np.prod([self.kernel_size[i] for i in range(len(axes))])
        else:
            # normalization weight
            # z = self.pad(torch.ones_like(x), axes=axes)

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

        for i, axis in enumerate(axes):
            if self.kernel_size[i] <= 1:
                continue

            # if self.count_include_pad:
            #     z *= self.kernel_size[i]
            # else:
            #     z = z.unfold(axis, size=self.kernel_size[i], step=self.stride[i]).sum(dim=-1, keepdim=False)

            x = x.unfold(axis, size=self.kernel_size[i], step=self.stride[i]).sum(dim=-1, keepdim=False)

        x = x / z
        return x
