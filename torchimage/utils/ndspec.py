"""
General utilities for torchimage
"""
import numpy as np

from .ragged import get_ragged_ndarray, expand_ragged_ndarray, apply_ragged_ndarray
from itertools import product


class NdSpec:
    """
    N-dimensional specification data, which represents parameters
    like ``kernel_size`` and ``pad_width`` that can either be a
    list of scalars (with length equal to the number of dimensions)
    or a single scalar to be broadcast. Another example is the input
    1d kernel for separable convolution, which can be a single kernel
    (list of float) or a list of kernels with the same or different
    lengths.

    The motivation for NdSpec is to provide developers with a unified
    interface to store such data. The developer of a module has the idea
    of what an item should look like (e.g. pad_width should be a tuple of
    int), but they cannot foresee the user's input at runtime, which could
    be an item or a sequence of items. With scipy, numpy, and pytorch,
    they usually have to use functions such as ``normalize_sequence``
    to manually convert a scalar to a sequence with predefined length;
    they are also required to re-write input type checks for every function.

    With NdSpec, however, items are automatically processed such that
    ``NdSpec(data=item, item_shape=...)[i] == item`` for any integer ``i``.
    In other words, scalars can now be regarded as sequences as well.
    The only caveat is that the __iter__ method makes single-item NdSpec
    behave the same way as a length-1 list. So zip doesn't necessarily work,
    use staticmethod ``NdSpec.zip`` instead.

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
    >>> c1 = NdSpec([1, 2, 3, 4, 5], item_shape=[-1])
    >>> c1[2]
    [1, 2, 3, 4, 5]
    >>> c2 = NdSpec([[1, 2, 3], [4, 5]], item_shape=[-1])
    >>> c2[0]
    [1, 2, 3]
    >>> c2[1]
    [4, 5]
    >>> d = NdSpec("hello", item_shape=[2])
    >>> d[12]
    ('hello', 'hello')
    >>> NdSpec("hello", item_shape=[2, -1])
    NdSpec(data=(('hello',), ('hello',)), item_shape=(2, -1))


    A more detailed introduction:

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
    In functionalities like `padding`,
    parameters like ``pad_width``, ``constant_values``, and ``stat_length``
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
    def __init__(self, data, *, item_shape=(), index_shape=None):
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

        index_shape : tuple or None
            Expected index shape of an item. Default is None (unknown,
            automatically infer from data and item_shape).

            Use -1 in the index shape to denote that the length of that
            axis doesn't matter.
        """
        if isinstance(data, NdSpec):  # constructor from NdSpec; shallow copy
            self.data = data.data
            self.shape = data.shape
            self.index_pos = data.index_pos
            return

        data, shape = get_ragged_ndarray(data, strict=True)

        if index_shape is None:
            item_shape = tuple(int(x) for x in item_shape)
            data, shape = expand_ragged_ndarray(data, old_shape=shape, new_shape=item_shape)
            self.index_pos = len(shape) - len(item_shape)
        else:  # ignore item shape
            if not all(length == -1 or shape[i] == length for i, length in enumerate(index_shape)):
                raise ValueError(f"Data {data} cannot have index shape {index_shape}")
            self.index_pos = len(index_shape)

        self.data = data
        self.shape = shape

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def item_shape(self):
        return self.shape[self.index_pos:]

    @property
    def index_shape(self):
        return self.shape[:self.index_pos]

    @property
    def is_item(self):
        return self.index_pos == 0

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

    def __getitem__(self, key):
        if self.is_item:
            return self.data
        else:
            if isinstance(key, tuple):
                if len(key) > len(self.index_shape):
                    raise IndexError(f"Too many indices {key} for ndim={self.ndim} and item_ndim={len(self.item_shape)}")

                ret = self.data
                for i in key:
                    ret = ret[i]
                return ret
            # key is int
            return self.data[key]

    def __iter__(self):
        if self.is_item:  # just one item
            return (x for x in [self.data])
        else:
            return (self[i] for i in range(len(self)))

    def __repr__(self):
        return f"NdSpec(data={self.data}, item_shape={self.item_shape}, index_shape={self.index_shape})"

    def map(self, func):
        """
        Apply a function element-wise to every item in self.

        Parameters
        ----------
        func : Callable
            The function that applies to every item in self.

        Returns
        -------
        ret : NdSpec
            Output from applying the function element-wise; index shape is maintained
        """
        return NdSpec(apply_ragged_ndarray(self.data, func=func, depth=self.ndim - len(self.item_shape)), index_shape=self.index_shape)

    def starmap(self, func):
        """
        Apply a function element-wise to every item in self.

        Instead of ``func(item)`` (as in ``NdSpec.map``),
        this method essentially applies ``func(*item)``.
        It will cause an error if item is not iterable.

        Parameters
        ----------
        func : Callable
            The function that applies to every item in self.

        Returns
        -------
        ret : NdSpec
            Output from applying the function element-wise; index shape is maintained
        """
        def func_star(item):
            return func(*item)
        return self.map(func_star)

    @staticmethod
    def zip(*args):
        """
        Zip a sequence of NdSpec such that each new item
        is an ordered tuple of each corresponding item in args.

        For example, ``zip(NdSpec([1, 2, 3]), NdSpec(4))`` will
        be ``NdSpec([(1, 4), (2, 4), (3, 4)], item_shape=(2,))``

        Parameters
        ----------
        args : NdSpec

        Returns
        -------
        ret : NdSpec
        """
        return NdSpec.apply(lambda *a: tuple(a), *args)

    @staticmethod
    def agg_index_shape(*args):
        """
        Get the aggregate index shape of input NdSpecs.

        The shape of an NdSpec is ``index_shape + item_shape``,
        where index shape is usually empty (when the NdSpec
        is an item) or (n,) where n is an integer (when the NdSpec
        is a list of items). Higher-order index shapes are supported
        but rare.

        Note that items are not necessarily scalars, they
        can also be a tuple of object with different shapes.

        Parameters
        ----------
        args : NdSpec
            Input NdSpec objects

        Returns
        -------
        index_shape : tuple of int
            Aggregate index shape of the NdSpecs (as if they
            were zipped)

        Raises
        ------
        ValueError
            When the input NdSpecs are not broadcastable
            (have different nonempty index shapes)
        """
        if not args:
            raise ValueError("Requires at least 1 input NdSpec")

        if not all(isinstance(a, NdSpec) for a in args):
            raise ValueError("Input arguments must have type NdSpec")

        index_shape_set = set(a.index_shape for a in args)
        if len(index_shape_set) == 1:
            return next(iter(index_shape_set))  # only 1 index shape
        elif len(index_shape_set) == 2 and () in index_shape_set:
            index_shape_set.remove(())
            return next(iter(index_shape_set))  # 2 shapes, 1 trivial
        else:
            raise ValueError(f"Input NdSpecs cannot broadcast: {args} have different index shapes {index_shape_set}")

    @staticmethod
    def apply(func, *args):
        """
        Generate a new NdSpec, whose every item is the result of
        applying the function item-wise on the sequence of items
        from input NdSpecs in corresponding positions.

        Parameters
        ----------
        func : function
            The function to be applied item-wise.

            It has signature ``func(item_1, item_2, ...)`` where
            the sequence of items has the same length as `args`.

        args : NdSpec
            An indefinite number of NdSpecs with the same index shape

        Returns
        -------
        nds : NdSpec
            An aggregate NdSpec with the same index shape
        """
        # equivalent to zip().starmap(func)
        index_shape = NdSpec.agg_index_shape(*args)

        if index_shape == ():  # all args are items
            data = func(*[a.data for a in args])
        else:
            data = np.empty(index_shape, dtype=object)
            for i in product(*map(range, index_shape)):
                data[i] = func(*[a[i] for a in args])
            data = data.tolist()

        return NdSpec(data, index_shape=index_shape)



