"""
Private utility functions for padding
"""
import torch
from math import ceil


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


def same_padding_width(kernel_size, stride=1, in_size=None):
    """
    Calculate the padding width before and after a certain axis
    using "same padding" method.

    When stride is 1, input size at that axis doesn't matter
    and the output tensor will have the same shape as the
    input tensor, hence the name "same padding".

    When stride is greater than 1, same padding can be intuitively
    described as "letting the kernel cover every element of
    the original tensor, while making padding width before and
    after the axis roughly the same." (unlike valid padding,
    which doesn't pad at all and the last pixels will be ignored
    if input tensor's side length doesn't match kernel size and stride)

    This convention is taken from a TensorFlow documentation page
    which no longer exists.

    Parameters
    ----------
    kernel_size : int
        The convolution kernel size at that axis

    stride : int
        The convolution stride at that axis. Default: 1.

    in_size : int
        The side length of the input tensor at that axis.

        Can be None if stride is 1.

    Returns
    -------
    pad_before, pad_after : int
        The number of padded elements required by same padding
        before and after the axis.
    """
    if stride == 1:
        pad_total = kernel_size - 1
    else:
        if in_size is None:
            raise ValueError(f"when stride={stride} instead of 1, in_size is required for same padding width")
        # expected output tensor size at axis with same padding
        out_size = ceil(in_size / stride)  # in_size == outsize with stride=1
        pad_total = (out_size - 1) * stride + kernel_size - in_size

    pad_total = max(pad_total, 0)
    pad_before = pad_total // 2
    pad_after = pad_total - pad_before
    return pad_before, pad_after


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

