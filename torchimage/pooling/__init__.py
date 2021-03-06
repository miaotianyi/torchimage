"""
We use pooling to refer to convolution-like operations that don't have
learnable parameters, such as average pooling and gaussian pooling.

In torchimage, what makes pooling different from filtering is that
a filter layer usually consists of its pooling layer counterpart
with stride 1, after a "same" padding layer that ensures the output
and the input will have the same shape. We separate filtering and pooling
so that users can have greater freedom and the code becomes
easier to maintain.
"""

from .base import BasePoolNd, SeparablePoolNd
from .uniform import AvgPoolNd
from .gaussian import GaussianPoolNd


__all__ = [
    "BasePoolNd", "SeparablePoolNd",
    "GaussianPoolNd", "AvgPoolNd",
]
