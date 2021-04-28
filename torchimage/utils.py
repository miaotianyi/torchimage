"""
General utilities for torchimage
"""
import numpy as np


class NdSpec:
    """
    N-dimensional specification data, which represents parameters
    like ``kernel_size`` that can either be a tuple or a scalar to
    be broadcast.

    In neural networks and n-dimensional arrays, sometimes
    the same set of parameters have different values for different
    shape dimensions (depth, height, width, etc. NOT channel or batch
    dimension). For example, ``kernel_size``, ``dilation``, and
    ``stride`` can either be a tuple of integers whose length is the
    same as the number of shape dimensions, or an integer (scalar)
    in which case the same value broadcasts to all shape dimensions.

    We call the returned parameter for each dimension an *item*.
    In the case above, each item is a scalar (integer). The
    corresponding NdSpec is defined to have *item shape* ``()``
    because the item itself is a scalar (not an iterable).
    In functionalities like :py:mod:`padding`,
    parameters like ``pad``, ``constant_values``, and ``stat_length``
    have ``item_shape == (2,)`` since padding width, etc. can be different
    before and after the source tensor.
    If only a scalar ``c`` is supplied, it should first be broadcast
    to length 2 (e.g. ``(c, c)``) and then returned for query.

    Indeed, we assume that each item has the same shape, as no
    counterexample has been found in practice.

    **NdSpec enforces a nested, right-justified, left-to-right storage format.**

    numpy pad is not right-justified: it requires every dimension to have item input, even if it's just (0, 0)
    otherwise it will raise an error

    torch pad is not nested; it's also right-to-left


    """
    def __init__(self, data, item_shape):
        self.item_shape = tuple(int(x) for x in item_shape)
        self.data = np.array(data)

        if self.data.shape == self.item_shape:
            self.is_item = True  # data is item
        elif self.data.ndim < len(self.item_shape):
            # requires extra broadcasting
            reps = list(self.item_shape)
            reps[-self.data.ndim:] = [1] * self.data.ndim
            self.data = np.tile(self.data, reps)

            assert self.data.shape == self.item_shape
            self.is_item = True  # data is item now
        else:
            assert self.data.shape[1:] == self.item_shape
            self.is_item = False

    def __call__(self, axis: int, ndim=None):
        if self.is_item:
            return self.data.tolist()

        if axis < 0:
            return self.data[axis].tolist()

        if ndim is None:
            raise ValueError("Only negative indices are allowed when ndim is unknown and is_item is False")

        new_axis = axis - (ndim - self.data.shape[0])  # ignore leading axes (batch, channel, etc.)
        return self.data[new_axis].tolist()

    def format_numpy(self, ndim):
        pass

    def format_torch(self):
        pass

