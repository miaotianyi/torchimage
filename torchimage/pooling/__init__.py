"""
We use pooling to refer to convolution-like operations that don't have
learnable parameters, such as average pooling and gaussian pooling.

In torchimage, what makes pooling different from filtering is that
a filter layer usually consists of its pooling layer counterpart
with stride 1, after a "same" padding layer that ensures the output
and the input will have the same shape. We separate filtering and pooling
so that users can have greater freedom and the code becomes
easier to maintain.

TODO: deprecate the following comment
    almost all other tensor operations cause copying.
Some may argue that in separable filters,
padding each dimension before immediately convolving can save some
computational costs, but we decide that copying a tensor for padding
isn't worth it, considering that the padding elements are relatively
few in comparison with the large amount of original tensor elements.
"""

from .gaussian import GaussianPool
from .generic import GenericPool2d, MedianPool2d, QuantilePool2d

__all__ = [
    "GaussianPool",
    "GenericPool2d", "MedianPool2d", "QuantilePool2d"
]
