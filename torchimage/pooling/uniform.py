import numpy as np

from .base import SeparablePoolNd
from ..utils import NdSpec


class AveragePoolNd(SeparablePoolNd):
    # TODO: this average method doesn't have an option to count_include_pad
    #   For example, with kernel_size=3, we should allow the corner pixel
    #   to be the average of 4 instead of 9 nearby pixels

    @staticmethod
    def _get_kernel(kernel_size):
        return NdSpec(kernel_size).map(lambda ks: (np.ones(ks) / ks).tolist())

    def __init__(self, kernel_size, stride=None):
        super().__init__(kernel=AveragePoolNd._get_kernel(kernel_size=kernel_size), stride=stride)
