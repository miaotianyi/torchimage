import numpy as np
import torch
from torch.nn import functional as F


def any_conv_1d(x: torch.Tensor, w: torch.Tensor, *, dim: int, stride: int, dilation: int):
    """
    Perform 1d convolution on any axis of a tensor.

    This operation is a more versatile version of ``x.unfold(...) @ w``
    because it can change ``dilation``.

    However, this method doesn't avoid the large memory consumption of unfold operation.

    Parameters
    ----------
    x : torch.Tensor

    w : torch.Tensor

    dim : int

    stride : int

    dilation : int

    Returns
    -------

    """
    big_shape = list(x.shape)

    if dim == -1 or dim == x.ndim - 1:
        x = x.view(-1, 1, x.shape[-1])  # batch, in_channels, input width
        w = w.view(1, 1, -1)  # out_channels, in_channels/groups, input kernel width
        x = F.conv1d(x, w, bias=None, stride=stride, padding=0, dilation=dilation, groups=1)
    else:
        x = x.view(np.prod(x.shape[:dim], dtype=int), 1, x.shape[dim], -1)
        w = w.view(1, 1, -1, 1)  # out_channels, in_channels/groups, input kernel height, input kernel width
        x = F.conv2d(x, w, bias=None, stride=(stride, 1), padding=0, dilation=(dilation, 1), groups=1)

    big_shape[dim] = -1
    x = x.view(*big_shape)
    return x

