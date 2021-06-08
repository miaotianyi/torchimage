import torch
from torch import nn
from ..padding.generic_pad import Padder
from ..utils import NdSpec
from ..utils.validation import check_axes, check_stride


def move_tensor(x: torch.Tensor, *, dtype, device):
    if torch.is_tensor(x):
        return x.clone().detach().to(dtype=dtype, device=device)
    return torch.tensor(x, dtype=dtype, device=device)


class BasePoolNd:
    kernel_size: NdSpec

    stride: NdSpec

    def __init__(self, *, same_padder: Padder = None):
        """
        Parameters
        ----------
        same_padder : Padder
            Pad the input tensor with this same padder.

            Because it uses the same-padding convention to automatically
            adjust pad_width, you can leave the padder's pad_width at any
            value since it'll be overridden anyway.

            See `same_padding_width` for the most complete definition of
            same padding. We use the convention from scipy instead of
            tensorflow (more padding before)

            It is a deliberate choice to only allow same padding
            to become a parameter of a pooling module: The widths
            of same padding depend on kernel size, stride, and the
            input tensor's shape, so it can only be inferred during
            runtime.

            If you wish to use custom padding widths at every axis,
            define a padding layer before the convolution module.
            (Some call this full padding)
            If you wish to not use padding at all, just leave
            ``same_padder=None`` unchanged and don't use any padding.
            (Some call this valid padding)

        """
        self.same_padder = same_padder

    def pad(self, x: torch.Tensor, axes):
        """
        Use the bound same padder to pad the input
        tensor before performing actual convolution

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        axes : int, slice, tuple of int
            Axes to convolve (processed to be nonnegative integers)

        Returns
        -------
        x : torch.Tensor
            padded tensor
        """
        axes = check_axes(x, axes)
        if self.same_padder is not None:
            padder = self.same_padder.to_same(kernel_size=self.kernel_size, stride=self.stride,
                                              in_shape=[x.shape[a] for a in axes])
            x = padder.forward(x, axes=axes)
        return x

    def read_stride(self, stride):
        """
        Read an input stride after ``self.kernel_size`` has been initialized.

        If stride is None at any axis, it will be the same as kernel size.

        This is a constant function (doesn't modify self).

        Parameters
        ----------
        stride: None, int, list, NdSpec
            Input stride

        Returns
        -------
        stride : NdSpec
            Processed stride
        """
        stride = NdSpec(stride, item_shape=[]).map(check_stride)
        stride = NdSpec.apply(lambda s, ks: ks if s is None else s, stride, self.kernel_size)
        return stride


class SeparablePoolNd(BasePoolNd):  # (nn.Module):
    def __init__(self, kernel=(), stride=None, *, same_padder: Padder = None):
        """
        N-dimensional separable pooling

        In torchimage, we use pooling to refer to convolution with
        no learnable parameters.

        Parameters
        ----------
        kernel : list, tuple, NdSpec
            Convolution kernel at each axis.

            Usually represented by a 1d float array (list, tuple, array),
            or a sequence thereof (NdSpec, list of list, etc.)

            If kernel is empty at any axis, that axis will be ignored
            in the forward method.

        stride : None, int, or a sequence thereof
            Convolution stride for each axis.

            If ``None``, it is the same as the kernel size at that axis.

        same_padder : Padder
        """
        super(SeparablePoolNd, self).__init__(same_padder=same_padder)
        self.kernel = NdSpec(kernel, item_shape=[-1])
        self.kernel_size = NdSpec(self.kernel.map(len), item_shape=[])

        self.stride = self.read_stride(stride)

        self._align_params()

    def _align_params(self):
        self.ndim = NdSpec.agg_index_len(self.kernel, self.stride, allowed_index_ndim=(0, 1))

    def forward(self, x: torch.Tensor, axes=slice(2, None)):
        """
        Perform separable pooling on a tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        axes : None, int, slice, tuple of int
            An ordered list of axes to be processed. Default: slice(2, None)

            Axes can be repeated. If ``None``, it will be all the axes
            from axis 0 to the last axis.

            The default ``slice(2, None)`` assumes that the first 2
            axes are batch (N) and channel (C) dimensions.

        Returns
        -------
        x : torch.Tensor
            Output tensor after pooling
        """
        axes = check_axes(x, axes)
        # move kernel to corresponding device/dtype
        kernel = self.kernel.map(lambda k: move_tensor(k, dtype=x.dtype, device=x.device))

        # initialize same padder
        x = self.pad(x, axes=axes)

        for i, axis in enumerate(axes):
            if self.kernel_size[i] == 0:
                continue

            x = x.unfold(axis, size=self.kernel_size[i], step=self.stride[i]) @ kernel[i]
        return x

    def to_filter(self, padder: Padder = None):
        """
        Modify this pooling module in-place, so that
        the stride is 1 and a same or valid padder is supplied.

        In torchimage, filtering is a special subset of pooling
        that has ``stride=1`` and (usually) same padding.
        (Considering that torch has not implemented a general
        method to perform dilated unfold on a tensor, dilation=1
        is the default.)

        Parameters
        ----------
        padder : Padder
            A same padder for this pooling module in the forward
            stage. A not-None padder will override
            self.same_padder. So if self.same_padder and padder
            are both None, valid padding will be used.

        Returns
        -------
        self : SeparablePoolNd
            A modified self
        """
        if padder is not None:
            self.same_padder = padder

        self.stride = NdSpec(1, item_shape=[])
        self._align_params()
        return self

