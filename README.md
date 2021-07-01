# torchimage
[![Documentation Status](https://readthedocs.org/projects/torchimage/badge/?version=latest)](https://torchimage.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.com/miaotianyi/torchimage.svg?branch=main)](https://travis-ci.com/miaotianyi/torchimage)

PyTorch utilities for classic image processing and evaluation

[Documentation](https://torchimage.readthedocs.org)

Highlights:

1. ``from torchimage.utils import NdSpec`` introduces a ``NdSpec`` class.
   It automatically converts user input in parameters like ``kernel_size``,
   which can be a scalar or list of scalars (unknown until runtime),
   into a wrapped class with a ``__getitem__`` method.

2. ``from torchimage.padding import Padder`` offers the most versatile Pytorch
   padding functionalities to date, including
   zeros, constant, replicate, smooth, circular, periodize, symmetric,
   reflect, antisymmetric, odd_reflect, odd_symmetric, linear_ramp,
   maximum, mean, median, minimum, empty.
   
3. ``from torchimage.pooling import GaussianPoolNd, AvgPoolNd`` offers
   faster gaussian pooling and average pooling at arbitrary dimensions.
   (It is approximately 10x faster than PyTorch thanks to separable filtering.)

4. ``from torchimage.metrics import SSIM, MS_SSIM, PSNR, MSE`` brings
   differentiable metrics in image processing.
   
5. ``from torchimage.filtering import Sobel, Prewitt, Farid, Scharr, GaussianGrad, Laplace, LaplacianOfGaussian``
   offers a range of edge detection modules.
   
6. ``from torchimage.filtering import UnsharpMask`` implements image
   sharpening algorithm

7. ``from torchimage.misc import poly1d`` calculates fast single-variable
   polynomial with constant coefficients.

Motivation:

1. We might want to use some classic image processing algorithms
   together with a neural network (e.g. loss, preprocessing).
   By making them *differentiable*, torchimage allows for their
   seamless integration into your PyTorch pipeline, which also
   avoids frequently moving tensors between devices.

2. Some of these algorithms don't previously have a Python
   version at all.
   
3. Many algorithms cannot process high-dimensional data
   (such as image batches or video batches) and therefore do not
   scale.

4. Being written in PyTorch means that torchimage can be
   run on both CPU and GPU using the same code, eliminating
   any need to adapt code implementation.
   
5. One of the main reasons for inconsistent behaviors in
   different image processing libraries is that some functions
   are implemented slightly differently, the most prominent
   example being padding (extending border of a signal). Not
   all packages have the same set of padding options; to make
   things worse, the same name may refer to different things.
   torchimage therefore aims to provide a standard for all
   methods to compare with.
   
