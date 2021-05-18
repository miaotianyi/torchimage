"""
General utilities for torchimage
"""
from .ragged import get_ragged_ndarray, expand_ragged_ndarray


class NdSpec:
    """
    N-dimensional specification data, which represents parameters
    like ``kernel_size`` that can either be a tuple or a scalar to
    be broadcast.

    The input can also be a filter (list of float) or a list of filters,
    which couldn't be known until runtime.

    NdSpec is implemented with ragged ndarray (see `ragged.py`), so
    it inherits most of the syntax and semantics.

    Examples:

    >>> a = NdSpec(1453, item_shape=[])
    >>> len(a)
    0
    >>> a[0]
    1453
    >>> a[-81]
    1453
    >>> a[12]
    1453
    >>> b = NdSpec([1, 2, 3, 4, 5], item_shape=[])
    >>> b
    NdSpec(data=[1, 2, 3, 4, 5], item_shape=())
    >>> len(b)
    5
    >>> b[0]
    1
    >>> b[4]
    5
    >>> b[-1]
    5
    >>> b[5]
    IndexError: list index out of range
    >>> c = NdSpec([1, 2, 3, 4, 5], item_shape=[-1])
    >>> c[2]
    [1, 2, 3, 4, 5]
    >>> d = NdSpec("hello", item_shape=[2])
    >>> d[12]
    ('hello', 'hello')

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

    In separable convolution, for example, we might want to pass
    in a list of filters (1d arrays of floats). Since filters can have
    different sizes, each filter is regarded as a Python object so
    ``item_shape == (-1)``.

    **NdSpec enforces a nested, right-justified, left-to-right storage
    format** when a list of items is passed.
    Take padding width (number of padded elements before and after the
    original tensor at a certain axis) for example, NdSpec format should be
    ``((before_{axis}, after_{axis}), (before_{axis+1}, after_{axis+1}),
    ..., (before_{n-2}, after_{n-2}), (before_{n-1}, after_{n-1}))``.

    - nested: Every element in the data tuple is an item.
    - right-justified: The last item in data corresponds to the last axis
      in the tensor of interest. Other axes are aligned correspondingly.
    - left-to-right: The axis with smaller index comes first.

    The motivation is best understood through comparison.
    Numpy padding width format is
    ``((before_0, after_0), (before_1, after_1), ..., (before_{n-1}, after_{n-1}))``.
    PyTorch padding width format is
    ``(before_{n-1}, after_{n-1}, before_{n-2}, after_{n-2}, ...,
    before_{axis}, after_{axis})``,
    such that axes less than ``axis`` are not padded.
    Numpy padding width format is nested and left-to-right
    (which makes it easier to read and index), but it forces users to
    supply specification for every axis in an n-dimensional array.
    This is cumbersome for (N, C, [D, ]H, W) data tensors, where
    batch and channel dimensions usually don't need padding.
    PyTorch padding width format is non-nested (ravel), right-to-left,
    and right-justified. Ravel makes it harder to access padding width
    for any specific axis without using pointer arithmetic. Right-to-left
    is sometimes counterintuitive. It also doesn't support automatic
    broadcasting (e.g. ``c -> [c, c] -> [[c, c]] * n`` where ``n``
    is the number of padded dimensions). Therefore, torchimage
    aims to get the best of both worlds with this new format.
    """
    def __init__(self, data, item_shape=()):
        """
        Parameters
        ----------
        data : scalar or array-like
            A partial item, an item, or an iterable of items,
            where each item is the specification for an axis.

            A partial item (smaller than item_shape) will be
            broadcast to item_shape, an item is kept as is,
            and an iterable of items is stored in a nested,
            right-justified, and left-to-right format.

        item_shape : tuple
            Expected shape of an item, i.e. ``np.array(item).shape``.

            If the item's length at a certain axis is uncertain
            or if different items may have different lengths at that
            axis, use ``-1`` to denote the item shape there.
            e.g. if the input data is a kernel (1d array)
            or a list of kernels with potentially different lengths.
        """
        self.item_shape = tuple(int(x) for x in item_shape)
        data, shape = get_ragged_ndarray(data, strict=True)
        data, shape = expand_ragged_ndarray(data, old_shape=shape, new_shape=self.item_shape)

        self.data = data
        self.ndim = len(shape)
        self.shape = shape
        self.is_item = self.ndim == len(item_shape)

    def __len__(self):
        """
        The length of NdSpec;
        it's zero if the NdSpec only has 1 item and can be broadcast
        to arbitrary length.
        """
        if self.is_item:
            return 0
        else:
            return self.shape[0]

    def __getitem__(self, *args):
        if self.is_item:
            return self.data
        else:
            if len(args) > self.ndim - len(self.item_shape):
                raise IndexError(f"Too many indices {args} for ndim={self.ndim} and item_ndim={len(self.item_shape)}")

            ret = self.data
            for i in args:
                ret = ret[i]
            return ret

    def __iter__(self):
        if self.is_item:  # just one item
            return (x for x in [self.data])
        else:
            return (self[i] for i in range(len(self)))

    def __repr__(self):
        return f"NdSpec(data={self.data}, item_shape={self.item_shape})"
