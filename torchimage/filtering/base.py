import torch
from ..padding import GenericPadNd
from ..pooling.base import SeparablePoolNd
from .utils import _same_padding_pair
import warnings


class SeparableFilterNd(SeparablePoolNd):
    def __init__(self, kernel: object):
        """
        N-dimensional separable filtering

        Parameters
        ----------
        kernel : 1d float array, or a sequence thereof
            Filter kernel at each axis.

        """
        warnings.warn("SeparableFilterNd is deprecated, please use decorator pool_to_filter instead", DeprecationWarning)
        super(SeparableFilterNd, self).__init__(kernel=kernel, stride=1)

    def forward(self, x: torch.Tensor, axes=None, same=True, padder: GenericPadNd = None):
        """
        Perform separable filtering on a tensor.

        Parameters
        ----------
        x : torch.Tensor

        axes : None or sequence

        same : bool
            If True, implements same padding.

        padder : GenericPadNd

        Returns
        -------
        y : torch.Tensor
        """
        if same and padder is not None:
            # same padding
            same_pad_width = self.kernel_size.map(_same_padding_pair)
            padder = GenericPadNd(pad_width=same_pad_width,
                                  mode=padder.mode.data,
                                  constant_values=padder.constant_values.data,
                                  end_values=padder.end_values.data,
                                  stat_length=padder.stat_length.data)

        return super(SeparableFilterNd, self).forward(x, axes=axes, padder=padder)

