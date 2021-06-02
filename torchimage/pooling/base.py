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
            if padder is not None:
                x = padder.pad_axis(x, axis=axis)

            if self.stride[i] is None:
                stride = self.kernel_size[i]
            else:
                stride = self.stride[i]

            x = x.unfold(axis, size=self.kernel_size[i], step=stride) @ kernel[i]
        return x


class LaplacePoolNd(nn.Module):
    def __init__(self, kernel=(1, -2, 1), stride=1):
        """
        "Separable" n-dimensional pooling inspired by
        generic Laplace filters.

        Unlike SeparablePoolNd, which sequentially applies
        1d convolution on previous output at each axis,
        LaplacePoolNd simultaneously applies the kernel to
        each axis, generating n output tensors in parallel;
        these output tensors are then added to obtain a
        final output.
        Therefore, the equivalent kernel of SeparablePoolNd
        is the outer product of each 1d kernel; the equivalent
        kernel of LaplacePoolNd is the sum of 1d kernels
        (after they are expanded to the total number of dimensions).

        For instance, 1d Laplace is [1, -2, 1] and 2d Laplace is
        [[0, 1, 0],
         [1, -4, 1],
         [0, 1, 0]]
        """
        super().__init__()
        self.kernel = NdSpec(kernel, item_shape=[-1])
        self.kernel_size = NdSpec(self.kernel.map(len), item_shape=[])
        self.stride = NdSpec(stride, item_shape=[])

    def forward(self, x: torch.Tensor, axes=None, padder: GenericPadNd = None):
        """
        This requires every 1d pooling to return a tensor of exactly the
        same shape, so we recommend stride=1 and same=True.

        Parameters
        ----------
        x
        axes
        padder

        Returns
        -------

        """
        axes = check_axes(x, axes)
        # move kernel to corresponding device/dtype
        kernel = NdSpec(self.kernel.map(lambda k: torch.tensor(k, dtype=x.dtype, device=x.device)), item_shape=[-1])

        ret = None

        for i, axis in enumerate(axes):
            if padder is not None:
                y = padder.pad_axis(x, axis=axis)
            else:
                y = x

            if self.stride[i] is None:
                stride = self.kernel_size[i]
            else:
                stride = self.stride[i]

            y = y.unfold(axis, size=self.kernel_size[i], step=stride) @ kernel[i]

            if ret is None:
                ret = y
            else:
                ret += y

        return ret

