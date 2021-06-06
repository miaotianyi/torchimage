

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


def check_stride(stride):
    if stride is None:
        return stride

    try:
        stride = int(stride)
    except TypeError:
        raise ValueError(f"stride must be integer, got {stride} of type {type(stride)} instead")

    if stride < 1:
        raise ValueError(f"stride must be positive, got {stride} instead")
    return stride


def check_pad_width(pad_before, pad_after):
    try:
        pad_before, pad_after = int(pad_before), int(pad_after)
    except TypeError:
        raise ValueError(f"padding width must be integer, got {pad_before} and {pad_after} instead")

    if pad_before < 0 or pad_after < 0:
        raise ValueError(f"padding width must be nonnegative, got {pad_before} and {pad_after} instead")
    return pad_before, pad_after


