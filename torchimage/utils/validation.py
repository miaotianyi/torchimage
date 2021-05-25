

def check_axes(x, axes):
    """
    Checks if a list of axes is valid for a tensor or ndarray
    Parameters
    ----------
    x : torch.Tensor or np.ndarray

    axes : sequence of int

    Returns
    -------
    axes : tuple of int
        Processed tuple of int axis indices
    """
    if axes is None:
        axes = tuple(range(x.ndim))
    elif isinstance(axes, slice):
        axes = tuple(range(x.ndim)[axes])
    else:
        axes = tuple(int(a) for a in axes)
        assert all(0 <= a <= x.ndim for a in axes)
        assert len(set(axes)) == len(axes)  # no repeated axes
    return axes
