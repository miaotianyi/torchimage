"""

"""

import torch

from ..utils import NdSpec


def separable_convolve(x: torch.Tensor, stride, kernel, axes, padder=None):
    kernel = NdSpec(kernel)
    stride = NdSpec(stride)
    axes = NdSpec(axes)


