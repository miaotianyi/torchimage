"""
We use pooling to refer to convolution operations that don't have learnable
parameters, such as average pooling and gaussian pooling.
"""

from .gaussian import GaussianPool

__all__ = [
    "GaussianPool"
]
