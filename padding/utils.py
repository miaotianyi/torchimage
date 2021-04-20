"""
Private utility functions for padding
"""
import torch


def _make_idx(*args, dim, ndim):
    """
    Make an index that slices exactly along a specified dimension.
    e.g. [:, ... :, slice(*args), :, ..., :]

    This helper function is similar to numpy's ``_slice_at_axis``.

    Parameters
    ----------
    *args : int or None
        constructor arguments for the slice object at target axis

    dim : int
        target axis

    ndim : int
        total number of axes

    Returns
    -------
    idx : tuple of slice
        Can be used to index np.ndarray and torch.Tensor
    """
    return (slice(None), ) * dim + (slice(*args), ) + (slice(None), ) * (ndim - dim - 1)


def _modify_idx(*args, idx, dim):
    """
    Make an index that slices a specified dimension while keeping the slices
    for other dimensions the same.

    Parameters
    ----------
    *args : int or None
        constructor arguments for the slice object at target axis

    idx : tuple of slice
        tuple of slices in the original region of interest

    dim : int
        target axis

    Returns
    -------
    new_idx : tuple of slice
        New tuple of slices with dimension dim substituted by slice(*args)

        Can be used to index np.ndarray and torch.Tensor
    """
    new_idx = list(idx)
    new_idx[dim] = slice(*args)
    return tuple(new_idx)


def _check_padding(x, pad):
    assert isinstance(x, torch.Tensor)
    assert hasattr(pad, "__len__")  # pad is list-like
    assert len(pad) % 2 == 0
    assert len(pad) // 2 <= x.ndim
    assert all(hasattr(p, "__index__") and p >= 0 for p in pad)  # is nonnegative int

