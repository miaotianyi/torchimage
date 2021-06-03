"""
Miscellaneous utilities for PyTorch

Many of these functions have not been implemented in PyTorch
"""

from .outer import outer
from .polynomial import poly1d

__all__ = [
    "outer",
    "poly1d"
]
