import numpy as np
import torch
from torch import nn
from ..misc import _repeat_align

from .base import SeparablePoolNd
from ..utils import NdSpec

# use gaussian kernel from scipy
from scipy.ndimage.filters import _gaussian_kernel1d

import warnings


def gaussian_kernel_1d(kernel_size, sigma, order):
    """
    generate a 1-dimensional Gaussian kernel given kernel_size and sigma.

    Multi-dimensional gaussian can be created by repeatedly obtaining outer products from 1-d Gaussian kernels.
    But when the application is Gaussian filtering (pooling) implemented through convolution,
    separable convolution is much more efficient.

    This function uses the utility function from scipy.ndimage
    to generate gaussian kernels

    Parameters
    ----------
    kernel_size : int
        length of the 1-d Gaussian kernel

        Please be aware that while even-length gaussian kernels can be generated,
        they are not well-defined and will cause a shift.

    sigma : float
        standard deviation of Gaussian kernel

    order : int
        An order of 0 corresponds to convolution with a Gaussian
        kernel. A positive order corresponds to convolution with
        that derivative of a Gaussian.


    Returns
    -------
    np.ndarray
        1-d Gaussian kernel of length kernel_size with standard deviation sigma
    """
    # require reverse
    # because scipy convolution (signal processing convention) flips the kernel
    # scipy convolution uses np.random, which automatically uses float64
    # device and dtype will be manually adjusted later in inference stage
    kernel = _gaussian_kernel1d(sigma=sigma, order=order, radius=kernel_size // 2)[::-1]
    if len(kernel) == kernel_size:
        return kernel.tolist()
    else:
        kernel = kernel[:kernel_size]  # convert odd-sized kernel to even-sized; remove last element
        # re-normalize doesn't apply to higher orders
        if order == 0:
            kernel /= kernel.sum()
        return kernel.tolist()


class GaussianPoolNd(SeparablePoolNd):
    @staticmethod
    def _get_kernel(kernel_size, sigma, order):
        kernel_params = NdSpec.zip(NdSpec(kernel_size), NdSpec(sigma), NdSpec(order))
        return kernel_params.starmap(lambda ks, s, o: gaussian_kernel_1d(kernel_size=ks, sigma=s, order=o))

    def __init__(self, kernel_size, sigma, order=0, stride=None):
        super(GaussianPoolNd, self).__init__(
            kernel=GaussianPoolNd._get_kernel(kernel_size=kernel_size, sigma=sigma, order=order), stride=stride)


class GaussianPool(nn.Module):
    """
    Applies n-dimensional Gaussian pooling over an input signal.

    Exploiting the fact that Gaussian kernel is separable, we can reduce the computational complexity
    from O(Nk^d) to O(Nkd), where N is the number of elements in the input tensor, k is kernel size,
    and d is the number of dimensions to convolve.

    Methods
    -------
    gaussian_kernel_1d(kernel_size, sigma)

    Attributes
    ----------
    ndim : int or None
        Number of dimensions for Gaussian kernel to convolve over.

        If None, then the Gaussian pooling module can be applied to arbitrary-dimension input tensor.

    kernel_size : int or sequence of ints

    sigma : float or sequence of floats

    stride : int or sequence of ints

    kernel : torch.Tensor

    kernel_list : list of torch.Tensor
    """
    @staticmethod
    def gaussian_kernel_1d(kernel_size, sigma):
        """
        generate a 1-dimensional Gaussian kernel given kernel_size and sigma.

        Multi-dimensional gaussian can be created by repeatedly obtaining outer products from 1-d Gaussian kernels.
        But when the application is Gaussian filtering (pooling) implemented through convolution,
        separable convolution is much more efficient.

        Parameters
        ----------
        kernel_size : int
            length of the 1-d Gaussian kernel

        sigma : float
            standard deviation of Gaussian kernel

        Returns
        -------
        torch.Tensor
            1-d Gaussian kernel of length kernel_size with standard deviation sigma
        """
        x = torch.arange(kernel_size) - kernel_size // 2
        x = x.double()  # use double for higher precision
        # device and dtype will be manually adjusted later during inference
        # no need for sqrt(2pi) because the kernel is normalized anyway
        x = 1. / sigma * torch.exp(-x ** 2 / (2 * sigma ** 2))
        return x / x.sum()  # normalize

    def __init__(self, kernel_size, sigma, stride):
        """
        Parameters
        ----------
        kernel_size : int or sequence of ints
            Sizes for Gaussian kernel.

        sigma : float or sequence of floats
            Standard deviation for Gaussian kernel. The standard deviations of the Gaussian filter are given
            for each axis as a sequence, or as a single number, in which case it is equal for all axes.

        stride : int or sequence of ints
            Strides for convolution
        """
        super(GaussianPool, self).__init__()

        warnings.warn("Deprecation Warning: Old gaussian pooling (doesn't use NdSpec) will be removed soon", DeprecationWarning)

        if stride is None:
            stride = kernel_size

        (self.kernel_size, self.sigma, self.stride), self.ndim = _repeat_align(kernel_size, sigma, stride)

        if self.ndim is None:
            self.kernel = GaussianPool.gaussian_kernel_1d(self.kernel_size, self.sigma)
            self.kernel_list = None
        else:
            self.kernel = None
            self.kernel_list = [GaussianPool.gaussian_kernel_1d(k, s) for k, s in zip(self.kernel_size, self.sigma)]

    def forward(self, x: torch.Tensor, dim=2):
        """
        Convolution is applied to all dimensions starting from ``dim`` (inclusive).

        Recall that 1d (NCL), 2d (NCHW), and 3d (NCDHW) tensors in deep learning all adopt the convention
        of using the first 2 axes as batch and channel dimensions. Therefore dim=2 is set as default.
        """
        if self.ndim is None:  # all scalars
            kernel = self.kernel.to(device=x.device, dtype=x.dtype)
            for i in range(x.ndim)[dim:]:
                x = x.unfold(i, size=self.kernel_size, step=self.stride) @ kernel
        else:
            assert len(x.shape[dim:]) == self.ndim
            for i, ks, kernel, stride in zip(range(x.ndim)[dim:], self.kernel_size, self.kernel_list, self.stride):
                # i: axis/dimension index for x
                kernel = kernel.to(device=x.device, dtype=x.dtype)
                x = x.unfold(i, size=ks, step=stride) @ kernel
        return x
