from torch import nn
from torch.nn import functional as F

from ..pooling import MedianPool2d, QuantilePool2d, GaussianPool


def _same_padding_pair(kernel_size):
    """
    Shape of padding required to keep input and output shape the same, with stride=dilation=1

    Parameters
    ----------
    kernel_size : int
        Size of convolution or pooling kernel

    Returns
    -------
    pad_beg : int
        Number of padding at the beginning of the axis.

    pad_end : int
        Number of padding at the end of the axis
    """
    pad_beg = kernel_size // 2
    if kernel_size % 2 == 1:  # odd kernel size, most common
        pad_end = pad_beg
    else:
        # this is somewhat arbitrary: padding less at the end
        # follow the same convention as scipy.ndimage
        pad_end = pad_beg - 1
    return pad_beg, pad_end


class GenericFilter2d(nn.Module):
    """
    Perform 2-D filtering on an image.

    The filter is created by adding a padding layer before the user-given pooling layer,
    whose stride and dilation are both set to 1 (they are both customizable parameters
    in a generic 2d pooling layer). The padding layer is defined such that input and output
    of a filter (as a whole) should have the same shape.

    For example, a median filter is composed of a padding layer followed by a median pooling layer.
    """
    def __init__(self, pooling, kernel_size, padding_mode="reflect", padding_value=0.0, **kwargs):
        """
        Parameters
        ----------
        pooling : str or Object
            The pooling layer to use in the filter.

            You can pass keyword arguments to use common pooling layers.
            Currently supported keyword arguments: ``median``, ``max``, ``mean``, ``quantile``, ``gaussian``

            Warning: ``quantile`` and ``median`` may consume too much memory. Proceed with caution.

            It's also possible to pass a customized pooling layer object.
            However, ``pooling`` must be a callable with ``kernel_size`` attribute.

        kernel_size : int or pair of int
            Size of the convolving kernel

        padding_mode : str
            ``"constant"``, ``"reflect"``, ``"replicate"`` or ``"circular"``. Default: ``"constant"``

        padding_value : float
            fill value for ``"constant"`` padding. Default: ``0``

        **kwargs
            Other parameters for the pooling layer if ``pooling`` is a keyword str.
        """
        super(GenericFilter2d, self).__init__()

        if pooling == "median":  # initialize median pooling
            assert kernel_size is not None
            self.kernel_size = kernel_size
            self.pooling_layer = MedianPool2d(self.kernel_size, stride=1, dilation=1)
        elif pooling in ("max", "maximum"):  # max pooling
            assert kernel_size is not None
            self.kernel_size = kernel_size
            self.pooling_layer = nn.MaxPool2d(self.kernel_size, stride=1, padding=0, dilation=1)
        elif pooling in ("mean", "uniform", "average"):  # mean/average pooling
            assert kernel_size is not None
            self.kernel_size = kernel_size
            self.pooling_layer = nn.AvgPool2d(self.kernel_size, stride=1, padding=0)
        elif pooling in ("quantile", "percentile"):
            assert kernel_size is not None
            self.kernel_size = kernel_size
            assert "q" in kwargs
            self.pooling_layer = QuantilePool2d(self.kernel_size, q=kwargs["q"], stride=1, dilation=1)
        elif pooling == "gaussian":
            assert kernel_size is not None
            self.kernel_size = kernel_size
            assert "sigma" in kwargs
            self.pooling_layer = GaussianPool(self.kernel_size, sigma=kwargs["sigma"], stride=1)
        elif hasattr(pooling, "__call__") and hasattr(pooling, "kernel_size"):
            # initialize with pre-defined pooling layer
            self.pooling_layer = pooling
            self.kernel_size = pooling.kernel_size
        else:
            raise ValueError(f"Unknown pooling {pooling} with type {type(pooling)}."
                             f"pooling must be a keyword str or nn.Module")

        assert padding_mode in ("constant", "reflect", "replicate", "circular")
        self.padding_mode = padding_mode

        self.padding_value = padding_value

        # generate padding based on kernel size
        # with stride=1, dilation=1, padding should make input and output size the same
        if hasattr(self.kernel_size, "__index__"):
            self.padding = _same_padding_pair(self.kernel_size) * 2
        else:
            assert hasattr(self.kernel_size, "__len__")  # has length
            assert len(self.kernel_size) == 2
            assert all(hasattr(k, "__index__") for k in self.kernel_size)
            self.padding = ()
            for k in self.kernel_size:
                self.padding = _same_padding_pair(k) + self.padding
                # note the order of addition
                # in F.pad function, padding is ordered like beg[-1], end[-1], beg[-2], end[-2]...
                # the last axis gets padded first
                # whereas in the usual NC[D]HW representation of images,
                # we state kernel sizes from left to right.

    def forward(self, x):
        if self.padding_mode == "constant":
            x = F.pad(x, self.padding, mode=self.padding_mode, value=self.padding_value)
        else:
            x = F.pad(x, self.padding, mode=self.padding_mode)
        return self.pooling_layer(x)

