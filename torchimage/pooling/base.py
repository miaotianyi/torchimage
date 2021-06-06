import torch
from torch import nn
from ..padding.generic_pad import Padder
from ..utils import NdSpec
from ..utils.validation import check_axes, check_stride


def move_tensor(x: torch.Tensor, *, dtype, device):
    if torch.is_tensor(x):
        return x.clone().detach().to(dtype=dtype, device=device)
    return torch.tensor(x, dtype=dtype, device=device)


class SeparablePoolNd: #(nn.Module):
    def __init__(self, kernel=(), stride=None, padder: Padder = None, *, same=False, separable_pad=False):
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

        padder : Padder
            Pad the input tensor with this padder.

        same : bool
            Whether to override pad_width of padder and use same padding.
            Default: False

            See `same_padding_width` for the most complete definition of
            same padding.

            If True, the ``pad_width`` argument in padder will
            be overridden during forward phase, so you may leave it to
            default when constructing the padder in the first place).

            If False, padder will be used as-is. So if you wish to use
            valid padding (in image filtering terminology, that means
            no padding), simply put ``padder=None``. For full padding
            (return the entire processed image, especially when customized
            padding makes the image shape larger), use any padder
            you want with ``same=False``.

            Because same padding needs to be calculated based on kernel size,
            stride, and input size, it cannot become a parameter of
            the padder.

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
        stride = NdSpec(stride, item_shape=[]).map(check_stride)
        self.stride = NdSpec.apply(lambda s, ks: ks if s is None else s,
                                   stride, self.kernel_size)

        # attributes related to padder
        self.padder = padder
        self.same = bool(same)
        self.separable_pad = bool(separable_pad)

        self.ndim = None
        self._align_params()

    def _align_params(self):
        index_shape = NdSpec.agg_index_shape(self.kernel, self.stride)
        assert len(index_shape) <= 1
        self.ndim = index_shape[0] if index_shape else 0

    def forward(self, x: torch.Tensor, axes=slice(2, None)):
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

        Returns
        -------
        x : torch.Tensor
            Output tensor after pooling
        """
        axes = check_axes(x, axes)
        # move kernel to corresponding device/dtype
        kernel = self.kernel.map(lambda k: move_tensor(k, dtype=x.dtype, device=x.device))

        # initialize same padder
        if self.same and self.padder is not None:
            padder = self.padder.to_same(kernel_size=self.kernel_size, stride=self.stride,
                                         in_shape=[x.shape[a] for a in axes])
        else:
            padder = self.padder

        if padder is not None and not self.separable_pad:  # pad all at once
            x = padder(x, axes=axes)

        for i, axis in enumerate(axes):
            if self.kernel_size[i] == 0:
                continue

            if padder is not None and self.separable_pad:
                x = padder.pad_axis(x, axis=axis)

            x = x.unfold(axis, size=self.kernel_size[i], step=self.stride[i]) @ kernel[i]
        return x

    def to_filter(self, same=True):
        """
        Modify this pooling module in-place, so that
        the stride is 1 and same is True.

        In torchimage, filtering is a special subset of pooling
        that has ``stride=1`` and (usually) same padding.
        (Considering that torch has not implemented a general
        method to perform dilated unfold on a tensor, dilation=1
        is the default.)

        Parameters
        ----------
        same : bool
            Whether to use same padding. Default: True

        Returns
        -------
        self : SeparablePoolNd
            A modified self
        """
        self.same = same
        self.stride = NdSpec(1, item_shape=[])
        self._align_params()
        return self

