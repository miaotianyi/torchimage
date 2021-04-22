"""
We use pooling to refer to convolution operations that don't have learnable
parameters, such as average pooling and gaussian pooling.
"""

from .gaussian import GaussianPool
from .generic import GenericPool2d, MedianPool2d, QuantilePool2d

__all__ = [
    "GaussianPool",
    "GenericPool2d", "MedianPool2d", "QuantilePool2d"
]
