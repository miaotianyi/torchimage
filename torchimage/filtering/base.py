import torch
from ..padding import GenericPadNd
from ..pooling.base import SeparablePoolNd
from .utils import _same_padding_pair


class SeparableFilterNd(SeparablePoolNd):
    def __init__(self, kernel: object):
        """
        N-dimensional separable filtering

        In torchimage, filtering is a special subset of pooling
        that has ``stride=1`` and (usually) same padding.
        (Considering that torch has not implemented a general
        method to perform dilated unfold on a tensor, dilation=1
        is the default.)
        In same padding, the input and output shapes are the same.

        Parameters
        ----------
        kernel : 1d float array, or a sequence thereof
            Filter kernel at each axis.

        """
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

            When ``same=True``, the ``pad_width`` argument in padder will
            be overridden (so you may leave it to default when constructing
            the padder in the first place).

            If False, padder will be used as-is. So if you wish to use
            valid padding (in image filtering terminology, that means
            no padding), simply put ``padder=None``. For full padding
            (return the entire processed image, especially when customized
            padding makes the image shape larger), use any padder that
            you want.

        padder : GenericPadNd

        Returns
        -------
        y : torch.Tensor
        """
        if same and padder is not None:
            # same padding
            same_pad_width = self.kernel_size.apply(_same_padding_pair)
            padder = GenericPadNd(pad_width=same_pad_width,
                                  mode=padder.mode.data,
                                  constant_values=padder.constant_values.data,
                                  end_values=padder.end_values.data,
                                  stat_length=padder.stat_length.data)

        return super(SeparableFilterNd, self).forward(x, axes=axes, padder=padder)

