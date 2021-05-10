# torchimage
[![Documentation Status](https://readthedocs.org/projects/torchimage/badge/?version=latest)](https://torchimage.readthedocs.io/en/latest/?badge=latest)

PyTorch utilities for classic image processing and evaluation

[Documentation](https://torchimage.readthedocs.org)

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
   
