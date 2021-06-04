"""
Predefined edge detection filters

"convolution" refers to two different things in neural network literature and signal processing
"convolution" as in convolutional neural networks is actually cross-correlation in signal processing
Whereas the "convolution" in signal processing actually flips the kernel before calculating.


what is laplace?

ndimage specification
sobel, prewitt
array, axis, padding params
axis: The axis of input along which to calculate. Default is -1

algorithmically, ndimage sets all other axes from 0 to ndim-1 as smooth axes, and compute the direction
then. (cannot exclude axes)
ndimage DOES NOT support gradient magnitude
It also has no normalization option
Take sobel as example, [-1, 0, 1] are multiplied exactly as stated, whereas they should have been normalized


MATLAB filtering module
The end goal of torchimage is to replace MatLab's image processing toolbox


"""
import numpy as np
import torch
from torch import nn


from ..pooling.gaussian import gaussian_kernel_1d
from ..pooling.base import SeparablePoolNd
from .decorator import pool_to_filter
from ..padding import Padder
from ..utils import NdSpec
from ..utils.validation import check_axes


class EdgeDetector(nn.Module):
    def __init__(self, edge_kernel, smooth_kernel, normalize=False):
        super(EdgeDetector, self).__init__()
        self.edge_kernel = edge_kernel
        self.smooth_kernel = NdSpec(smooth_kernel, item_shape=[-1])
        if normalize:
            def _reweight(kernel):
                kernel = np.array(kernel)
                kernel = kernel / kernel.sum()
                return kernel.tolist()
            self.smooth_kernel = self.smooth_kernel.map(_reweight)

    def forward(self, x, mode, *, edge_axis=-1, smooth_axes=-2, axes=None,
                same=True, padder: Padder = Padder(mode="reflect"),
                epsilon=0.0, p=2):
        if mode == "component":
            return self.component(x, edge_axis=edge_axis, smooth_axes=smooth_axes, same=same, padder=padder)
        elif mode == "magnitude":
            return self.magnitude(x, axes=axes, same=same, padder=padder, epsilon=epsilon, p=p)
        else:
            raise ValueError(f"Edge detector mode must be component or magnitude, got {mode} instead")

    def component(self, x, edge_axis, smooth_axes, *, same=True, padder: Padder = Padder(mode="reflect")):
        edge_axis = check_axes(x, edge_axis)
        if len(edge_axis) != 1:
            raise ValueError(f"Only 1 edge axis is allowed, got {edge_axis} instead")
        smooth_axes = check_axes(x, smooth_axes)
        if edge_axis in smooth_axes:
            # happens when smooth_axes is initially None (all axes)
            smooth_axes = tuple(a for a in smooth_axes if a != edge_axis)

        SeparableFilterNd = pool_to_filter(SeparablePoolNd, same=same)

        edge_filter = SeparableFilterNd(kernel=self.edge_kernel)
        x = edge_filter(x, axes=edge_axis, padder=padder)
        smooth_filter = SeparableFilterNd(kernel=self.smooth_kernel)
        x = smooth_filter(x, axes=smooth_axes, padder=padder)
        return x

    def horizontal(self, x, *, same=True, padder: Padder = None):
        return self.component(x, edge_axis=-2, smooth_axes=-1, same=same, padder=padder)

    def vertical(self, x, *, same=True, padder: Padder = None):
        return self.component(x, edge_axis=-1, smooth_axes=-2, same=same, padder=padder)

    def magnitude(self, x, axes=None, *, same=True, padder: Padder = Padder(mode="reflect"), epsilon=0.0, p=2):
        axes = check_axes(x, axes)
        if len(axes) < 2:
            raise ValueError(f"Image gradient computation requires at least 2 axes, got {axes} instead")

        magnitude = None

        for i, edge_axis in enumerate(axes):
            # edge component at edge_axis
            c = self.component(x, edge_axis=edge_axis, smooth_axes=axes[:i]+axes[i+1:], same=same, padder=padder)
            if magnitude is None:
                magnitude = c ** p
            else:
                magnitude += c ** p

        return (magnitude + epsilon) ** (1/p)


class Sobel(EdgeDetector):
    def __init__(self, normalize=False):
        super().__init__(edge_kernel=(-1, 0, 1), smooth_kernel=(1, 2, 1), normalize=normalize)


class Prewitt(EdgeDetector):
    def __init__(self, normalize=False):
        super().__init__(edge_kernel=(-1, 0, 1), smooth_kernel=(1, 1, 1), normalize=normalize)


class Scharr(EdgeDetector):
    def __init__(self, normalize=False):
        super().__init__(edge_kernel=(-1, 0, 1), smooth_kernel=(3, 10, 3), normalize=normalize)


class Farid(EdgeDetector):
    def __init__(self, normalize=True):
        # These filter weights can be found in Farid & Simoncelli (2004),
        # Table 1 (3rd and 4th row). Additional decimal places were computed
        # using the code found at https://www.cs.dartmouth.edu/farid/

        # smooth kernel already normalized
        smooth = [0.0376593171958126, 0.249153396177344, 0.426374573253687, 0.249153396177344, 0.0376593171958126]
        edge = [-0.109603762960254, -0.276690988455557, 0, 0.276690988455557, 0.109603762960254]
        super().__init__(edge_kernel=edge, smooth_kernel=smooth)


class GaussianGrad(EdgeDetector):
    def __init__(self, kernel_size, sigma, edge_kernel_size=None, edge_sigma=None, normalize=True, edge_order=1):
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
        super().__init__(edge_kernel=edge, smooth_kernel=smooth)


class Laplace(EdgeDetector):
    """
    Edge detection with discrete Laplace operator

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

    This method is less general than scipy's generic_laplace,
    because we cannot customize a non-separable second derivative
    function.

    This requires every 1d pooling to return a tensor of exactly the
    same shape, so we recommend ``same=True``.
    """
    def __init__(self):
        super().__init__(edge_kernel=(1, -2, 1), smooth_kernel=(), normalize=False)

    def forward(self, x, mode="magnitude", *, edge_axis=-1, smooth_axes=-2, axes=None,
                same=True, padder: Padder = Padder(mode="reflect"),
                epsilon=0.0, p=1):
        return super().forward(x=x, mode=mode, edge_axis=edge_axis, smooth_axes=smooth_axes, axes=axes,
                               same=same, padder=padder, epsilon=epsilon, p=p)


class LaplacianOfGaussian(nn.Module):
    """
    The same as scipy.ndimage.gaussian_laplace
    """
    def __init__(self, kernel_size, sigma):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma

    def forward(self, x: torch.Tensor, axes=None, *, same=True, padder=Padder(mode="reflect")):
        gg = GaussianGrad(kernel_size=self.kernel_size, sigma=self.sigma, edge_order=2)
        return gg.magnitude(x, axes=axes, same=same, padder=padder, epsilon=0.0, p=1)
