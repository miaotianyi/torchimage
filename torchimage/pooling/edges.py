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

skimage specification


"""
from scipy import ndimage
print(ndimage.sobel([[0, 0, 1], [0, 0, 1], [0, 0, 1]]))

