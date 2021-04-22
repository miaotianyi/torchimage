"""
Generic 2d pooling and other derived subclasses
"""
import math
import torch
from torch import nn

from ..misc import _repeat_tuple, conv2d_shape_infer


class GenericPool2d(nn.Module):
    """
    Applies a customized 2D pooling over an input signal composed of several input planes.

    Shape
    -----
    - Input: :math:`(N, C, H_{in}, W_{in})`
    - Output:  :math:`(N, C, H_{out}, W_{out})`, where
        :math:`H_{out} =\lfloor \frac{ H_{in} - dilation[0] \times (kernel_size[0] - 1) - 1 }{stride[0]} + 1\rfloor`

        :math:`W_{out} =\lfloor \frac{ H_{in} - dilation[1] \times (kernel_size[1] - 1) - 1 }{stride[1]} + 1\rfloor`
    """

    def __init__(self, func, kernel_size, stride=None, dilation=1):
        """
        Parameters
        ----------
        func: torch.Tensor -> torch.Tensor
            Takes in a tensor of shape ``(n, c, k[0], k[1], L)`` and returns a tensor of shape ``(n, c, L)``.
            Essentially, func is a kernel-like function that reduces a kernel_size patch to a scalar.

        kernel_size : int or (int, int)
            the size of the window

        stride : int or (int, int)
            the stride of the window. Default: kernel_size

        dilation : int or (int, int)
            a parameter that controls the stride of elements within the neighborhood. Default: 1
        """
        super(GenericPool2d, self).__init__()

        if stride is None:
            stride = kernel_size

        self.unfold = nn.Unfold(kernel_size=kernel_size, dilation=dilation, padding=0, stride=stride)
        self.func = func

    def forward(self, x):
        n, c, h_in, w_in = x.shape
        x_unf = self.unfold(x)  # n, c*k[0]*k[1], L; k is kernel_size
        k0, k1 = _repeat_tuple(self.unfold.kernel_size, n=2)
        x_unf = x_unf.view(n, c, k0, k1, -1)  # n, c, k[0], k[1], L
        y = self.func(x_unf)  # n, c, L
        h_out, w_out = conv2d_shape_infer(
            h_in, w_in, kernel_size=self.unfold.kernel_size, stride=self.unfold.stride, dilation=self.unfold.dilation)
        y = y.view(n, c, h_out, w_out)
        return y


class MedianPool2d(GenericPool2d):
    @staticmethod
    def compute_median(x):
        """
        Perform median pooling on unfolded 2d input.

        Parameters
        ----------
        x : torch.Tensor
            Unfolded input tensor of shape ``(n, c, kernel_size[0], kernel_size[1], L)``,
            where ``L`` is the number of patches.

        Returns
        -------
        y : torch.Tensor
            Unfolded output tensor of shape ``(n, c, L)``,
            where ``L`` is the number of patches.

        Notes
        -----
        Given an even number of elements, median is not well-defined and ``torch.median`` always return the lower
        element. In this case, we use ``torch.quantile(y, 0.5)`` instead, which takes the average of the 2
        central elements
        """
        y = x.view(x.shape[0], x.shape[1], -1, x.shape[4])  # n, c, k0 * k1, L
        if y.shape[-2] % 2 == 1:  # odd shape
            y = torch.median(y, dim=-2, keepdim=False)[0]
        else:
            y = torch.quantile(y, q=0.5, dim=-2, keepdim=False)
        return y

    def __init__(self, kernel_size, stride=None, dilation=1):
        super().__init__(MedianPool2d.compute_median, kernel_size, stride, dilation)


class QuantilePool2d(GenericPool2d):
    def compute_quantile(self, x):
        """
        Perform q-quantile pooling on unfolded 2d input.

        Parameters
        ----------
        x : torch.Tensor
            Unfolded input tensor of shape ``(n, c, kernel_size[0], kernel_size[1], L)``,
            where ``L`` is the number of patches.

        Returns
        -------
        y : torch.Tensor
            Unfolded output tensor of shape ``(n, c, L)``,
            where ``L`` is the number of patches.

        Notes
        -----
        PyTorch's quantile function uses linear interpolation whenever necessary,
        which might behave differently from its counterparts in scipy and numpy.
        """
        y = x.view(x.shape[0], x.shape[1], -1, x.shape[4])  # n, c, k0 * k1, L
        y = torch.quantile(y, q=self.q, dim=-2, keepdim=False)
        return y

    def __init__(self, kernel_size, q, stride=None, dilation=1):
        super().__init__(self.compute_quantile, kernel_size, stride, dilation)
        self.q = q

