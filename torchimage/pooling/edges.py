"""
Predefined edge detection filters

"convolution" refers to two different things in neural network literature and signal processing
"convolution" as in convolutional neural networks is actually cross-correlation in signal processing
Whereas the "convolution" in signal processing actually flips the kernel before calculating.


TODO: what is laplace?

todo
can all be decomposed into 2 parts: smooth (like [1,2,1]) and edge (like [-1, 0, 1])

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
import torch
from torch import nn

from ..padding import GenericPadNd


def edge_magnitude(image, axes, padder: GenericPadNd = None):
    pass


class EdgeComponent(nn.Module):
    def __init__(self, edge_kernel, smooth_kernel):
        super().__init__()
        self.edge_kernel = edge_kernel
        self.smooth_kernel = smooth_kernel

    def forward(self, x, edge_axis, smooth_axes, same=True, padder: GenericPadNd = None):
        pass


class EdgeMagnitude(nn.Module):
    def __init__(self, edge_kernel, smooth_kernel):
        super().__init__()
        self.edge_kernel = edge_kernel
        self.smooth_kernel = smooth_kernel

    def forward(self, x, axes, same=True, padder: GenericPadNd = None):
        pass
