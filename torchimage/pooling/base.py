import torch
from torch import nn
from ..padding.generic_pad import Padder
from ..utils import NdSpec
from ..utils.validation import check_axes


class SeparablePoolNd(nn.Module):
    def __init__(self, kernel, stride=None, *, padder: Padder = None, separable_pad=False):
        """
        N-dimensional separable pooling

        In torchimage, we use pooling to refer to convolution with
        no learnable parameters.

        Parameters
        ----------
        kernel : 1d float array, or a sequence thereof
            Convolution kernel at each axis.

            Usually represented by a list/tuple/array of numbers.

            If kernel is empty at any axis, that axis will be ignored
            in the forward method.

        stride : None, int, or a sequence thereof
            Convolution stride for each axis.

            If ``None``, it is the same as the kernel size at that axis.

        padder : Padder
            Pad the input tensor with this padder.

        separable_pad : bool
            If True, pad each axis only before the separable convolution
            at that axis. Otherwise, pad the entire tensor before performing
            all convolution steps. Default: False.

            Setting ``separable_pad=True`` may lead to more intermediate
            tensors stored in the computational graph if any of the ancestors
            requires gradient. It is also slower unless the dimension
            is really high and the padding width far exceeds the input
            tensor size.
        """
        super(SeparablePoolNd, self).__init__()
        self.kernel = NdSpec(kernel, item_shape=[-1])
        self.kernel_size = NdSpec(self.kernel.map(len), item_shape=[])
        self.stride = NdSpec(stride, item_shape=[])

        self.padder = padder
        self.separable_pad = separable_pad

    def forward(self, x: torch.Tensor, axes=slice(2, None), padder: Padder = None):
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

        padder : Padder
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

