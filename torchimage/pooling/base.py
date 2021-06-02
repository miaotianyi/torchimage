import torch
from torch import nn
from ..padding.generic_pad import GenericPadNd
from ..utils import NdSpec
from ..utils.validation import check_axes


class SeparablePoolNd(nn.Module):
    def __init__(self, kernel, stride=None):
        """
        N-dimensional separable pooling

        In torchimage, we use pooling to refer to convolution with
        no learnable parameters.

        Parameters
        ----------
        kernel : 1d float array, or a sequence thereof
            Convolution kernel at each axis.

            Usually represented by a list/tuple/array of numbers.

            If kernel is ``()`` at any axis, that axis will be ignored
            in the forward method.

        stride : None, int, or a sequence thereof
            Convolution stride for each axis.

            If ``None``, it is the same as the kernel size at that axis.
        """
        super(SeparablePoolNd, self).__init__()
        self.kernel = NdSpec(kernel, item_shape=[-1])
        self.kernel_size = NdSpec(self.kernel.map(len), item_shape=[])
        self.stride = NdSpec(stride, item_shape=[])

    def forward(self, x: torch.Tensor, axes=None, padder: GenericPadNd = None):
        """
        Perform separable pooling on a tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        axes : None or list of int
            An ordered list of axes to be processed.

            Axes can be repeated. If ``None``, it will be all the axes
            from axis 0 to the last axis.

        padder : GenericPadNd
            A padder that performs axis-wise padding immediately before an axis
            is convolved (1d separable convolution).

        Returns
        -------
        x : torch.Tensor
            Output tensor after pooling
        """
        axes = check_axes(x, axes)
        # move kernel to corresponding device/dtype
        kernel = NdSpec(self.kernel.map(lambda k: torch.tensor(k, dtype=x.dtype, device=x.device)), item_shape=[-1])

        for i, axis in enumerate(axes):
            if len(kernel[i]) == 0:
                continue

            if padder is not None:
                x = padder.pad_axis(x, axis=axis)

            if self.stride[i] is None:
                stride = self.kernel_size[i]
            else:
                stride = self.stride[i]

            x = x.unfold(axis, size=self.kernel_size[i], step=stride) @ kernel[i]
        return x

