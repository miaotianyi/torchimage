

def check_axes(x, axes):
    """
    Checks if a list of axes is valid for a tensor or ndarray.

    If the input axis indices are not valid, a ValueError
    will be raised. Otherwise, the axes will be processed into
    a tuple of nonnegative integers

    Parameters
    ----------
    x : torch.Tensor or np.ndarray
        The input tensor. Axis indices cannot exceed its number
        of dimensions.

    axes : int, slice, or sequence of int
        An axis index, an integer slice, or a sequence of axis indices.

    Returns
    -------
    axes : tuple of int
        Processed tuple of int axis indices
    """
    if axes is None:
        axes = tuple(range(x.ndim))
    elif isinstance(axes, slice):
        axes = tuple(range(x.ndim)[axes])
    elif isinstance(axes, int):
        return tuple([axes])
    else:
        axes = tuple(int(a) if a >= 0 else x.ndim + int(a) for a in axes)
        assert all(0 <= a <= x.ndim for a in axes)
    return axes
