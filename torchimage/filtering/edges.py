"""
Predefined edge detection filters

"convolution" refers to two different things in neural network literature and signal processing
"convolution" as in convolutional neural networks is actually cross-correlation in signal processing
Whereas the "convolution" in signal processing actually flips the kernel before calculating.

algorithmically, ndimage sets all other axes from 0 to ndim-1 as smooth axes, and compute the direction
then. (cannot exclude axes)
ndimage DOES NOT support gradient magnitude
It also has no normalization option
Take sobel as example, [-1, 0, 1] are multiplied exactly as stated, whereas they should have been normalized


MATLAB filtering module
The end goal of torchimage is to replace MatLab's image processing toolbox

Comparison: skimage, scipy, MatLab, Kornia, OpenCV, gimp

"""
import numpy as np
import torch
from torch import nn

from ..pooling.gaussian import gaussian_kernel_1d
from ..pooling.base import SeparablePoolNd
from ..utils import NdSpec
from ..utils.validation import check_axes
from ..padding.utils import make_idx


class EdgeFilter(nn.Module):
    def __init__(self, edge_kernel, smooth_kernel, *, normalize=False, same_padder=None):
        super(EdgeFilter, self).__init__()

        # initialize kernels/weights
        self.edge_kernel = NdSpec(edge_kernel, item_shape=[-1])
        if self.edge_kernel.index_shape != ():
            raise ValueError(f"There can only be 1 edge kernel, got {edge_kernel} instead.")

        self.smooth_kernel = NdSpec(smooth_kernel, item_shape=[-1])
        if normalize:
            def _reweight(kernel):
                kernel = np.array(kernel)
                kernel = kernel / kernel.sum()
                return kernel.tolist()

            self.smooth_kernel = self.smooth_kernel.map(_reweight)

        # initialize padder
        self.same_padder = same_padder

        # initialize filters
        self.edge_filter = SeparablePoolNd(kernel=self.edge_kernel).to_filter(padder=self.same_padder)
        self.smooth_filter = SeparablePoolNd(kernel=self.smooth_kernel).to_filter(padder=self.same_padder)

    def _conv_edge(self, x: torch.Tensor, edge_axis: int):
        # some discrete kernels can be faster (slices instead of convolution)
        discrete_kernel = tuple(int(k) for k in self.edge_kernel[0])
        if discrete_kernel == (-1, 0, 1):
            x = self.edge_filter.pad(x, axes=edge_axis)
            return x[make_idx(2, None, dim=edge_axis, ndim=x.ndim)] - \
                x[make_idx(0, -2, dim=edge_axis, ndim=x.ndim)]
        elif discrete_kernel == (1, -2, 1):
            x = self.edge_filter.pad(x, axes=edge_axis)
            return x[make_idx(2, None, dim=edge_axis, ndim=x.ndim)] + \
                x[make_idx(0, -2, dim=edge_axis, ndim=x.ndim)] - \
                2 * x[make_idx(1, -1, dim=edge_axis, ndim=x.ndim)]
        return self.edge_filter.forward(x, axes=edge_axis)

    def _conv_smooth(self, x: torch.Tensor, smooth_axes):
        return self.smooth_filter.forward(x, axes=smooth_axes)

    def component(self, x, edge_axis, smooth_axes):
        edge_axis = check_axes(x, edge_axis)
        if len(edge_axis) != 1:
            raise ValueError(f"Only 1 edge axis is allowed, got {edge_axis} instead")
        edge_axis = edge_axis[0]

        smooth_axes = check_axes(x, smooth_axes)
        if edge_axis in smooth_axes:
            # happens when smooth_axes is initially None (all axes)
            smooth_axes = tuple(a for a in smooth_axes if a != edge_axis)

        x = self._conv_edge(x, edge_axis=edge_axis)
        x = self._conv_smooth(x, smooth_axes=smooth_axes)
        return x

    def all_components(self, x, axes=None):
        axes = check_axes(x, axes)
        if len(axes) < 2:
            raise ValueError(f"Image gradient computation requires at least 2 axes, got {axes} instead")

        component_list = []

        for edge_axis in axes:
            # edge component at edge_axis
            c = self.component(x, edge_axis=edge_axis, smooth_axes=axes)
            component_list.append(c)
        return component_list

    def horizontal(self, x):
        return self.component(x, edge_axis=-2, smooth_axes=-1)

    def vertical(self, x):
        return self.component(x, edge_axis=-1, smooth_axes=-2)

    def magnitude(self, x, axes=None, *, epsilon=0.0, p=2):
        # note that p isn't actually an L^p norm as it doesn't require positivity
        # it skips the absolute value step

        axes = check_axes(x, axes)
        if len(axes) < 2:
            raise ValueError(f"Image gradient computation requires at least 2 axes, got {axes} instead")

        magnitude = None

        for edge_axis in axes:
            # edge component at edge_axis
            c = self.component(x, edge_axis=edge_axis, smooth_axes=axes)

            # if p % 2 == 1:  # odd L^p norm -> need abs
            #     c = torch.abs(c)

            if magnitude is None:
                magnitude = c ** p if p != 1 else c
            else:
                magnitude += c ** p if p != 1 else c

        if epsilon != 0:
            magnitude += epsilon

        if p == 1:
            return magnitude
        else:
            return magnitude ** (1 / p)


class Sobel(EdgeFilter):
    def __init__(self, *, normalize=False, same_padder="reflect"):
        super().__init__(edge_kernel=(-1, 0, 1), smooth_kernel=(1, 2, 1), normalize=normalize, same_padder=same_padder)


class Prewitt(EdgeFilter):
    def __init__(self, *, normalize=False, same_padder="reflect"):
        super().__init__(edge_kernel=(-1, 0, 1), smooth_kernel=(1, 1, 1), normalize=normalize, same_padder=same_padder)


class Scharr(EdgeFilter):
    def __init__(self, *, normalize=False, same_padder="reflect"):
        super().__init__(edge_kernel=(-1, 0, 1), smooth_kernel=(3, 10, 3), normalize=normalize, same_padder=same_padder)


class Farid(EdgeFilter):
    def __init__(self, *, normalize=False, same_padder="reflect"):
        # These filter weights can be found in Farid & Simoncelli (2004),
        # Table 1 (3rd and 4th row). Additional decimal places were computed
        # using the code found at https://www.cs.dartmouth.edu/farid/

        # smooth kernel already normalized
        smooth = [0.0376593171958126, 0.249153396177344, 0.426374573253687, 0.249153396177344, 0.0376593171958126]
        edge = [-0.109603762960254, -0.276690988455557, 0, 0.276690988455557, 0.109603762960254]
        super().__init__(edge_kernel=edge, smooth_kernel=smooth, normalize=normalize, same_padder=same_padder)


class GaussianGrad(EdgeFilter):
    def __init__(self, kernel_size, sigma, edge_kernel_size=None, edge_sigma=None, normalize=True, edge_order=1,
                 same_padder="reflect"):
        kernel_size = NdSpec(kernel_size, item_shape=[])
        sigma = NdSpec(sigma, item_shape=[])
        smooth = NdSpec.apply(lambda ks, s: gaussian_kernel_1d(kernel_size=ks, sigma=s, order=0), kernel_size, sigma)

        if edge_kernel_size is None:
            assert kernel_size.is_item
            edge_kernel_size = kernel_size.data

        if edge_sigma is None:
            assert sigma.is_item
            edge_sigma = sigma.data
        edge = gaussian_kernel_1d(kernel_size=edge_kernel_size, sigma=edge_sigma, order=edge_order)
        self.edge_order = edge_order
        super().__init__(edge_kernel=edge, smooth_kernel=smooth, normalize=normalize, same_padder=same_padder)


class Laplace(nn.Module):
    """
    Edge detection with discrete Laplace operator

    Unlike SeparablePoolNd, which sequentially applies
    1d convolution on previous output at each axis,
    Laplace simultaneously applies the kernel to
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

    This method is less general than scipy's generic_laplace,
    because we cannot customize a non-separable second derivative
    function.

    This requires every 1d pooling to return a tensor of exactly the
    same shape, so we recommend ``same=True``.
    """

    def __init__(self, *, same_padder="reflect"):
        super().__init__()
        self.ef = EdgeFilter(edge_kernel=(1, -2, 1), smooth_kernel=(), normalize=False, same_padder=same_padder)

    def forward(self, x: torch.Tensor, axes):
        return self.ef.magnitude(x, axes=axes, epsilon=0, p=1)


class LaplacianOfGaussian(nn.Module):
    """
    The same as scipy.ndimage.gaussian_laplace
    """

    def __init__(self, kernel_size, sigma, *, same_padder="reflect"):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.gg = GaussianGrad(kernel_size=self.kernel_size, sigma=self.sigma, edge_order=2, same_padder=same_padder)

    def forward(self, x: torch.Tensor, axes=None):
        return self.gg.magnitude(x, axes=axes, epsilon=0.0, p=1)
