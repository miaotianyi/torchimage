import numpy as np


def get_ndim(a):
    """
    Return the number of dimensions of an array.

    This method differs from ``np.ndim`` in that the array
    conversion (if invoked) uses ``dtype=object`` so ragged
    arrays will not cause an error.

    Parameters
    ----------
    a : array_like
        Input array. If it is not already an ndarray, a conversion
        without copy is attempted.

    Returns
    -------
    number_of_dimensions : int
        The number of dimensions in `a`. Scalars are zero-dimensional.
    """
    try:
        return a.ndim
    except AttributeError:
        return np.asarray(a, dtype=object).ndim


class RaggedArray:
    """
    Ragged multi-dimensional array.

    Ragged multi-dimensional arrays are a superclass to hyper-rectangular
    multi-dimensional arrays (like numpy ndarray), such that the shapes
    of array items may be different, e.g. ``[[1], [2, 3], [4, 5, 6]]``.
    However, the ndim of items at the same level must be the same.
    For example, ``[1, [2, 3]]`` is not a valid ragged ndarray but
    ``[[1], [2, 3]]`` is. In other words, items in the same array
    must be the same "type" in terms of nested array hierarchy.

    The shape of a ragged ndarray can be represented like the shape
    of a regular ndarray, where if array items have different lengths
    at a certain axis, the shape at that axis is defined as -1.
    """
    def __init__(self, data, is_scalar=lambda x: get_ndim(x) == 0):
        assert callable(is_scalar)
        self.is_scalar = is_scalar
        self.shape = RaggedArray.get_shape(data, is_scalar=self.is_scalar)

    @staticmethod
    def get_shape(data, is_scalar, auto_broadcast=False):
        """
        Recursive helper method for ragged array broadcasting
        and shape determination.

        Parameters
        ----------
        data : array_like

        is_scalar : Callable[[Any], bool]

        auto_broadcast : bool

        Returns
        -------
        data : array_like

        shape : tuple of int
        """
        if is_scalar(data):
            return ()

        length = len(data)
        if length == 0:  # empty array, no need to check
            return 0,  # singleton tuple

        shape_set = {RaggedArray.get_shape(item, is_scalar=is_scalar) for item in data}

        ndim_set = set(len(shape) for shape in shape_set)

        if len(ndim_set) > 1:
            raise ValueError(f"Items in array {data} have different ndim: {ndim_set}")

        ndim = next(iter(ndim_set))

        item_shape = [-1] * ndim

        for i in range(ndim):
            ss = {shape[i] for shape in shape_set}  # set of lengths at axis i
            if len(ss) > 1:
                item_shape[i] = -1
            else:
                item_shape[i] = next(iter(ss))

        return tuple([length] + item_shape)

    @staticmethod
    def expand(data, shape, is_scalar):
        pass


def broadcast_ragged(array):
    pass
    # return array, shape


def expand_ragged(array, source_shape, target_shape):
    pass

