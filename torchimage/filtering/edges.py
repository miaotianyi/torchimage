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

skimage filters




MATLAB filtering module
The end goal of torchimage is to replace MatLab's image processing toolbox


"""
import numpy as np
import torch
from torch import nn


from ..pooling.gaussian import gaussian_kernel_1d
from ..pooling.base import SeparablePoolNd
from .decorator import pool_to_filter
from ..padding import GenericPadNd
from ..utils import NdSpec
from ..utils.validation import check_axes


class EdgeDetector:
    def __init__(self, edge_kernel, smooth_kernel, normalize=False):
        self.edge_kernel = edge_kernel
        self.smooth_kernel = NdSpec(smooth_kernel, item_shape=[-1])
        if normalize:
            def _reweight(kernel):
                kernel = np.array(kernel)
                kernel = kernel / kernel.sum()
                return kernel.tolist()
            self.smooth_kernel = self.smooth_kernel.map(_reweight)

    def component(self, x, edge_axis, smooth_axes, *, same=True, padder: GenericPadNd = None):
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

    def horizontal(self, x, *, same=True, padder: GenericPadNd = None):
        return self.component(x, edge_axis=-2, smooth_axes=-1, same=same, padder=padder)

    def vertical(self, x, *, same=True, padder: GenericPadNd = None):
        return self.component(x, edge_axis=-1, smooth_axes=-2, same=same, padder=padder)

    def magnitude(self, x, axes=None, *, same=True, padder: GenericPadNd = None, epsilon=1e-8):
        axes = check_axes(x, axes)
        if len(axes) < 2:
            raise ValueError(f"Image gradient computation requires at least 2 axes, got {axes} instead")

        magnitude = None

        for i, edge_axis in enumerate(axes):
            # edge component at edge_axis
            c = self.component(x, edge_axis=edge_axis, smooth_axes=axes[:i]+axes[i+1:], same=same, padder=padder)
            if magnitude is None:
                magnitude = c ** 2
            else:
                magnitude += c ** 2

        return torch.sqrt(magnitude + epsilon)


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
    def __init__(self, normalize=False):
        # These filter weights can be found in Farid & Simoncelli (2004),
        # Table 1 (3rd and 4th row). Additional decimal places were computed
        # using the code found at https://www.cs.dartmouth.edu/farid/

        # smooth kernel already normalized
        smooth = [0.0376593171958126, 0.249153396177344, 0.426374573253687, 0.249153396177344, 0.0376593171958126]
        edge = [-0.109603762960254, -0.276690988455557, 0, 0.276690988455557, 0.109603762960254]
        super().__init__(edge_kernel=edge, smooth_kernel=smooth)


# class EdgeMagnitude(nn.Module):
#     def __init__(self, edge_component, epsilon=1e-8):
#         super().__init__()
#         self.component = edge_component
#         self.epsilon = epsilon  # small constant to prevent overflow
#
#     def forward(self, x, axes=None, same=True, padder: GenericPadNd = None):
#         axes = check_axes(x, axes)
#         if len(axes) < 2:
#             raise ValueError(f"Image gradient computation requires at least 2 axes, got {axes} instead")
#
#         magnitude = None
#
#         for i, edge_axis in enumerate(axes):
#             # edge component at edge_axis
#             c = self.component(x, edge_axis=edge_axis, smooth_axes=axes[:i]+axes[i+1:], same=same, padder=padder)
#             if magnitude is None:
#                 magnitude = c ** 2
#             else:
#                 magnitude += c ** 2
#
#         return torch.sqrt(magnitude + self.epsilon)




