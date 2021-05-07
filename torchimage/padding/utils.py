"""
Private utility functions for padding
"""
import torch


def make_idx(*args, dim, ndim):
    """
    Make an index that slices exactly along a specified dimension.
    e.g. [:, ... :, slice(*args), :, ..., :]

    This helper function is similar to numpy's ``_slice_at_axis``.

    Parameters
    ----------
    *args : tuple of int or None
        constructor arguments for the slice object at target axis

    dim : int
        target axis; can be negative or positive

    ndim : int
        total number of axes

    Returns
    -------
    idx : tuple of slice
        Can be used to index np.ndarray and torch.Tensor
    """
    if dim < 0:
        dim = ndim + dim
    return (slice(None), ) * dim + (slice(*args), ) + (slice(None), ) * (ndim - dim - 1)


def modify_idx(*args, idx, dim):
    """
    Make an index that slices a specified dimension while keeping the slices
    for other dimensions the same.

    Parameters
    ----------
    *args : tuple of int or None
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


def pad_width_format(padding, source="numpy", target="torch", ndim=None):
    """
    Convert between 2 padding width formats.

    This function converts ``pad`` from ``source`` format to ``target`` format.

    Padding width refers to the number of padded elements before and after
    the original tensor at a certain axis. Numpy and PyTorch have different
    formats to specify the padding widths. Because Numpy works with n-dimensional
    arrays while PyTorch more frequently works with (N, C, [D, ]H, W) data tensors.
    In the latter case, starting from the last dimension seems more intuitive.

    Numpy padding width format is
    ``((before_0, after_0), (before_1, after_1), ..., (before_{n-1}, after_{n-1}))``.

    PyTorch padding format is
    ``(before_{n-1}, after_{n-1}, before_{n-2}, after_{n-2}, ..., before_{dim}, after_{dim})``,
    such that before after ``dim`` is not padded.

    Parameters
    ----------
    padding : tuple of int, or tuple of tuple of int
        the input padding width format to convert

    source, target : str
        Format specification for padding width. Either "numpy" or "torch".

    ndim : int
        Number of dimensions in the tensor of interest.

        Only used when converting from torch to numpy format.

    Returns
    -------
    padding : tuple of int, or tuple of tuple of int
        the new padding width specification
    """
    if source == target:
        return padding

    if source == "numpy" and target == "torch":
        padding = tuple(tuple(x) for x in padding)
        padding = sum(padding[::-1], start=())
        return padding
    elif source == "torch" and target == "numpy":
        padding = tuple(padding)
        assert ndim is not None, "ndim not supplied for pad width conversion from torch to numpy"
        ndim_padded = len(padding) // 2
        return ((0, 0), ) * (ndim - ndim_padded) + tuple(padding[i:i + 2] for i in range(0, len(padding), 2))[::-1]
    else:
        raise ValueError(f"Unsupported pad width format conversion from {source} to {target}")

