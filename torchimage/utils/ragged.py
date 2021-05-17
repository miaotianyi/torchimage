import torch
import numpy as np


def get_ndim(a) -> int:
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


def get_shape(a):
    """
    Return the shape of an array.

    This method differs from ``np.shape`` in that the array
    conversion (if invoked) uses ``dtype=object`` so ragged
    arrays will not cause an error.

    Parameters
    ----------
    a : array_like
        Input array.

    Returns
    -------
    shape : tuple of ints
        The elements of the shape tuple give the lengths of the
        corresponding array dimensions.

    """
    try:
        result = a.shape
    except AttributeError:
        result = np.asarray(a, dtype=object).shape
    return tuple(result)


def get_ragged_ndarray(data, strict=True):
    """
    Recursive function for ragged array broadcasting and shape determination.

    Ragged multi-dimensional arrays are a superclass to hyper-rectangular
    multi-dimensional arrays (like numpy ndarray), such that the shapes
    of array items may be different, e.g. ``[[1], [2, 3], [4, 5, 6]]``.
    However, the ndim of items at the same level must be the same.
    For example, ``[1, [2, 3]]`` is not a valid ragged ndarray but
    ``[[1], [2, 3]]`` is. In other words, items in the same array
    must be the same "type" in terms of nested array hierarchy.

    It might be more intuitive to think of a ragged ndarray as a
    tree where every leaf has the same depth.

    We use the same logic as ``lambda x: get_ndim(x) == 0`` to determine
    if an object is a scalar (when converted to a numpy ndarray with
    ``dtype=object``, will it become a non-scalar array?)

    This data structure is designed to robustly handle all possible
    python data types, especially torch.Tensor and numpy.ndarray,
    but it's not optimized for time or space.

    Parameters
    ----------
    data : array_like
        Input ragged ndarray to be assessed.

    strict : bool
        If True, raises an error when sibling items different depths. Otherwise,
        attempt to broadcast items with smaller depth, e.g. [a, [b]] -> [[a], [b]].

        With tree terminology, the shallower leaves are "lifted" to the maximum
        depth and new trivial nodes are filled in between.

    Returns
    -------
    data : array_like
        The same ragged ndarray. It is modified only when ``strict=False``
        and sibling items have different lengths (automatically broadcast).

        The broadcasting process uses tuple type for its immutability.

    shape : tuple of int
        The shape of the ragged ndarray; represented similarly to that of
        a regular ndarray.

        If array items have different lengths at a certain axis, the shape
        at that axis is defined as -1.
    """
    if torch.is_tensor(data):
        # torch Tensor cannot be ragged
        # it also cannot take on non-numeric values
        return data, tuple(data.shape)

    if isinstance(data, np.ndarray):
        if np.issubdtype(data.dtype, np.number) or np.issubdtype(data.dtype, np.bool_):
            # cannot be ragged with numeric types
            return data, tuple(data.shape)

    shape = get_shape(data)

    if len(shape) == 0 or 0 in shape:
        # scalar, ndim == 0 (ndim = len(shape)), or:
        # empty array with nonempty shape (e.g. shape=[0, 10])
        return data, shape

    item_list = []
    item_shape_list = []
    for item in data:
        item, item_shape = get_ragged_ndarray(item, strict=strict)
        item_list.append(item)
        item_shape_list.append(item_shape)

    ndim_set = set(len(shape) for shape in item_shape_list)

    if len(ndim_set) > 1:
        if strict:
            raise ValueError(f"Items in array {data} have different ndim: {ndim_set}")
        # otherwise auto broadcast
        ndim = max(ndim_set)
        for i, (item, item_shape) in enumerate(zip(item_list, item_shape_list)):
            for j in range(ndim - len(item_shape)):
                item = (item, )  # broadcast add depth
            item_list[i] = item
        data = tuple(item_list)
        item_shape_list = [(1, ) * (ndim - len(s)) + s for s in item_shape_list]
    else:
        ndim = next(iter(ndim_set))

    item_shape = [-1] * ndim

    for i in range(ndim):
        length_i_set = {s[i] for s in item_shape_list}  # set of lengths at axis i
        if len(length_i_set) > 1:
            item_shape[i] = -1
        else:
            item_shape[i] = next(iter(length_i_set))

    return data, tuple([len(data)] + item_shape)


def recursive_expand(arr, target_shape):
    # shape has the same depth as arr
    if not target_shape:  # reaches leaf node
        return arr, target_shape

    if torch.is_tensor(arr):
        result = arr.expand(*target_shape)
        return result, tuple(result.shape)

    if isinstance(arr, np.ndarray):
        if np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.bool_):
            target_shape = [old if new == -1 else new for old, new in zip(arr.shape, target_shape)]
            result = np.broadcast_to(arr, shape=target_shape)
            return result, tuple(target_shape)

    # actual shape before broadcast
    shape = get_shape(arr)

    if len(shape) == 0 or 0 in shape:
        # scalar, ndim == 0 (ndim = len(shape)), or:
        # empty array with nonempty shape (e.g. shape=[0, 10])
        return arr, target_shape

    item_list = []
    item_shape_list = []
    for item in arr:
        item, item_shape = recursive_expand(item, target_shape=target_shape[1:])
        item_list.append(item)
        item_shape_list.append(item_shape)

    arr = tuple(item_list)

    ndim_set = set(len(item_shape) for item_shape in item_shape_list)
    assert len(ndim_set) == 1
    item_ndim = next(iter(ndim_set))

    item_shape = [-1] * item_ndim

    for i in range(item_ndim):
        item_shape_set = {s[i] for s in item_shape_list}  # set of lengths at axis i
        if len(item_shape_set) > 1:
            item_shape[i] = -1
        else:
            item_shape[i] = next(iter(item_shape_set))

    # broadcast (expand) this dimension
    if target_shape[0] == -1 or target_shape[0] == len(arr):  # no need to expand
        return arr, tuple([len(arr)] + item_shape)
    elif target_shape[0] == 0:
        return (), tuple([0] + item_shape)
    elif len(arr) == 1:  # singleton dimension; shape[0] is positive int
        return arr * target_shape[0], tuple([target_shape[0]] + item_shape)
    else:
        raise ValueError(f"Non-singleton dimension {len(arr)} must match target length {target_shape[0]}")


def expand_ragged_ndarray(data, old_shape, new_shape):
    """
    Expand the singleton dimensions in a ragged ndarray.

    The input ragged ndarray of shape ``old_shape`` will be
    broadcast to ``new_shape``. This function is similar to
    ``torch.expand`` and ``numpy.broadcast_to``.

    Parameters
    ----------
    data : array_like
        Input ragged ndarray.

    old_shape : tuple of int
        Shape of the input ragged ndarray.

        data and old_shape can be obtained from `get_ragged_ndarray`.

    new_shape : tuple of int
        New shape that the ragged ndarray should be broadcast to.

        -1 at any axis in new shape means the original length
        at that dimension will remain the same.

    Returns
    -------
    data : array_like
        Expanded ragged ndarray

    final_shape : tuple of int
        The shape of the expanded ragged ndarray
    """
    old_shape = tuple(int(x) for x in old_shape)
    new_shape = tuple(int(x) for x in new_shape)

    old_ndim = len(old_shape)
    new_ndim = len(new_shape)

    if old_ndim < new_ndim:
        data, final_shape = recursive_expand(data, target_shape=new_shape[new_ndim-old_ndim:])
        for new_length in new_shape[:new_ndim-old_ndim][::-1]:
            if new_length == -1:  # no specification, add a trivial wrapper
                data = (data, )
                final_shape = (1, ) + final_shape
            else:
                data = (data, ) * new_length
                final_shape = (new_length, ) + final_shape
    elif old_ndim == new_ndim:
        data, final_shape = recursive_expand(data, target_shape=new_shape)
    else:  # old_ndim > new_ndim
        data, final_shape = recursive_expand(data, target_shape=(-1,) * (old_ndim - new_ndim) + new_shape)
    return data, final_shape
