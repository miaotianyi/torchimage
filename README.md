# torchimage
PyTorch utilities for classic image processing and evaluation

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
