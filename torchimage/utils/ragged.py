import torch
import numpy as np


def is_scalar(a) -> bool:
    """
    Tests if a python object is a scalar (instead of an array)

    Parameters
    ----------
    a : object
        Any object to be checked

    Returns
    -------
    bool
        Whether the input object is a scalar
    """
    if isinstance(a, (list, tuple)):
        return False
    if hasattr(a, "__array__") and hasattr(a, "__len__"):  # np.array(1) is scalar
        return False
    return True


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

    This data structure is designed to robustly handle all possible
    python data types, especially torch.Tensor and numpy.ndarray,
    but it's not optimized for time or space.

    Parameters
    ----------
    data : array_like
        Input ragged ndarray to be assessed.

    strict : bool
        If True, regards all items as leaf objects when sibling items different
        depths. Otherwise, attempt to broadcast items with smaller depth,
        e.g. [a, [b]] -> [[a], [b]].

        With tree terminology, in ``strict=False`` mode, the shallower leaves are
        "lifted" to the maximum depth and new trivial nodes are filled in between.
        Whereas in ``strict=True`` mode, the minimum depth is regarded as ground
        truth and deeper nodes are not counted.

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

    if is_scalar(data):  # scalar
        return data, ()

    if not data:  # empty array
        return data, (0, )

    item_list = []
    item_shape_list = []
    for item in data:
        item, item_shape = get_ragged_ndarray(item, strict=strict)
        item_list.append(item)
        item_shape_list.append(item_shape)

    ndim_set = set(len(shape) for shape in item_shape_list)

    if len(ndim_set) > 1:
        if strict:
            ndim = 0  # regarded as leaf
        else:
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


def _recursive_expand(arr, target_shape):
    # recursively expand the ragged ndarray according to target_shape
    # also returns a 3rd result (whether the array is indeed modified)

    # shape has the same depth as arr
    if not target_shape:  # reaches leaf node
        return arr, target_shape, False

    if torch.is_tensor(arr):
        result = arr.expand(*target_shape)
        return result, tuple(result.shape), result.shape != arr.shape

    if isinstance(arr, np.ndarray):
        if np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.bool_):
            target_shape = [old if new == -1 else new for old, new in zip(arr.shape, target_shape)]
            result = np.broadcast_to(arr, shape=target_shape)
            return result, tuple(target_shape), result.shape != arr.shape

    if is_scalar(arr):  # scalar
        return arr, (), False

    if not arr:  # empty array
        return arr, (0,), False

    item_list = []
    item_shape_list = []
    modified = False
    for item in arr:
        item, item_shape, item_modified = _recursive_expand(item, target_shape=target_shape[1:])
        item_list.append(item)
        item_shape_list.append(item_shape)
        modified = modified or item_modified

    if modified:
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
        return arr, tuple([len(arr)] + item_shape), modified
    elif target_shape[0] == 0:
        shape = tuple([0] + item_shape)
        return np.empty(shape), shape, True
    elif len(arr) == 1:  # singleton dimension; shape[0] is positive int
        return tuple(arr[0] for _ in range(target_shape[0])), tuple([target_shape[0]] + item_shape), True  # is modified
    else:
        raise ValueError(f"Non-singleton dimension {len(arr)} must match target length {target_shape[0]}")


def _check_zero_length(old_shape, new_shape):
    # error if try to broadcast 0 length to nonzero length
    for a1, a2 in zip(old_shape[::-1], new_shape[::-1]):  # right-justify using reversal
        if a1 == 0 and a2 > 0:
            raise ValueError(f"Cannot expand 0-length dimension to nonzero-length: "
                             f"(right-justified) {old_shape} to {new_shape}")


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

    # shortcut
    if old_shape == new_shape:
        return data, old_shape

    old_ndim = len(old_shape)
    new_ndim = len(new_shape)

    _check_zero_length(old_shape, new_shape)

    if old_ndim < new_ndim:
        # ignore modified
        data, final_shape, _ = _recursive_expand(data, target_shape=new_shape[new_ndim - old_ndim:])
        for new_length in new_shape[:new_ndim-old_ndim][::-1]:
            if new_length == -1:  # no specification, add a trivial wrapper
                data = (data, )
                final_shape = (1, ) + final_shape
            else:
                data = (data, ) * new_length
                final_shape = (new_length, ) + final_shape
    elif old_ndim == new_ndim:
        data, final_shape, _ = _recursive_expand(data, target_shape=new_shape)
    else:  # old_ndim > new_ndim
        data, final_shape, _ = _recursive_expand(data, target_shape=(-1,) * (old_ndim - new_ndim) + new_shape)
    return data, final_shape


def _recursive_apply(arr, func, depth):
    if depth == 0:  # regard this arr as item
        return func(arr)
    return tuple(_recursive_apply(sub, func=func, depth=depth-1) for sub in arr)


def apply_ragged_ndarray(arr, func, depth):
    if not hasattr(depth, "__index__") or depth < 0:
        raise ValueError(f"apply depth {depth} should be a nonnegative integer")
    return _recursive_apply(arr=arr, func=func, depth=depth)

