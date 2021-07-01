import numpy as np
import torch
from torch import nn

from .base import SeparablePoolNd
from ..utils import NdSpec


def _gaussian_kernel1d(sigma, order, radius):
    """
    Computes a 1-D Gaussian convolution kernel.

    This function is copied from scipy to avoid
    dependency issues when importing from scipy
    """
    if order < 0:
        raise ValueError('order must be non-negative')
    exponent_range = np.arange(order + 1)
    sigma2 = sigma * sigma
    x = np.arange(-radius, radius+1)
    phi_x = np.exp(-0.5 / sigma2 * x ** 2)
    phi_x = phi_x / phi_x.sum()

    if order == 0:
        return phi_x
    else:
        # f(x) = q(x) * phi(x) = q(x) * exp(p(x))
        # f'(x) = (q'(x) + q(x) * p'(x)) * phi(x)
        # p'(x) = -1 / sigma ** 2
        # Implement q'(x) + q(x) * p'(x) as a matrix operator and apply to the
        # coefficients of q(x)
        q = np.zeros(order + 1)
        q[0] = 1
        D = np.diag(exponent_range[1:], 1)  # D @ q(x) = q'(x)
        P = np.diag(np.ones(order)/-sigma2, -1)  # P @ q(x) = q(x) * p'(x)
        Q_deriv = D + P
        for _ in range(order):
            q = Q_deriv.dot(q)
        q = (x[:, None] ** exponent_range).dot(q)
        return q * phi_x


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
    """
    N-dimensional Gaussian Pooling

    This module is implemented using separable pooling.
    Recall that Gaussian kernel is separable, i.e.
    An n-dimensional Gaussian kernel is the outer product
    of n 1D Gaussian kernels. N-dimensional Gaussian
    convolution is equivalent to sequentially applying
    n 1D Gaussian convolutions sequentially to each
    axis of interest.
    We can thus reduce the computational complexity
    from O(Nk^d) to O(Nkd), where N is the number of elements
    in the input tensor, k is kernel size, and d is
    the number of dimensions to convolve.
    """
    @staticmethod
    def _get_kernel(kernel_size, sigma, order):
        kernel_params = NdSpec.zip(NdSpec(kernel_size), NdSpec(sigma), NdSpec(order))
        return kernel_params.starmap(lambda ks, s, o: gaussian_kernel_1d(kernel_size=ks, sigma=s, order=o))

    def __init__(self, kernel_size, sigma, order=0, stride=None, *, same_padder=None):
        """
        Parameters
        ----------
        kernel_size : int or sequence of int

        sigma : float or sequence of float

        order : int or sequence of int

        stride : int, None, or sequence of int
        """
        super(GaussianPoolNd, self).__init__(
            kernel=GaussianPoolNd._get_kernel(kernel_size=kernel_size, sigma=sigma, order=order),
            stride=stride,
            same_padder=same_padder
        )

