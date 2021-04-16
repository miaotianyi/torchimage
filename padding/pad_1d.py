"""
Pad a torch tensor along a certain dimension.

All 1-d padding utility functions share the same set of arguments.

**Important note**: To avoid re-copying the input tensor, these 1d padding
utility functions only accept an empty tensor that has the same shape as
the final padded output and copies the values of the input tensor at its center.
The function will return its input x after modifying it.

**Efficiency Warning**: Currently, PyTorch doesn't support returning negative-strided
views like [::-1]. torch.flip() is said to be computationally expensive due to copying
the data, which might put ``symmetric`` and ``reflect`` to a disadvantage.

They are for lower-level utility only. Do *NOT* expect
them to behave the same way as ``F.pad`` does.

Comparing with ``torch.nn.functional.pad``, we make the following improvements:

1. Symmetric padding is added

2. Higher-dimension padding
    To the date of this release, PyTorch has not implemented reflect (ndim >= 3+2),
    replicate (ndim >= 4+2), and circular (ndim >= 4+2) for high-dimensional
    tensors (the +2 refers to the initial batch and channel dimensions).

    This is achieved by decomposing n-dimensional padding to sequential padding
    along every dimension of interest.

3. Larger padding size
    Padding modes reflect and circular will cause PyTorch to fail when padding size
    is greater than the tensor's shape at a certain dimension.
    i.e. Padding value causes wrapping around more than once.

We design this padding package with these principles:

1. Maximal compatibility with PyTorch.
    We will implicitly use ``F.pad`` as much as we can.
    (For example, we do not implement zero and constant padding)
    We also adjust the names of the arguments to be compatible with PyTorch


Padding Mode
------------
The naming convention of padding modes is contested.
- Scipy's ``mirror`` is PyTorch's ``reflect``.
- Scipy's ``wrap`` is PyTorch's ``circular``.
- Scipy's ``reflect`` is implemented as ``symmetric`` here;
    it has no counterpart in PyTorch.

``"constant"`` - pads with a constant value
    ``p p p p | a b c d | p p p p`` where ``p`` is supplied by ``padding_value`` argument.

``"zeros"`` - pads with 0; a special case of constant padding
    ``0 0 0 0 | a b c d | 0 0 0 0``

``"symmetric"`` - extends signal by *mirroring* samples. Also known as half-sample symmetric
    ``d c b a | a b c d | d c b a``

``"reflect"`` - signal is extended by *reflecting* samples. This mode is also known as whole-sample symmetric
    ``d c b | a b c d | c b a``

``"replicate"`` - replicates the border pixels
    ``a a a a | a b c d | d d d d``

``"circular"`` - signal is treated as a periodic one
    ``a b c d | a b c d | a b c d``


Parameters
----------
x : torch.Tensor
    The input tensor to be padded.

    See the important note above. Let ``u`` be the original tensor,
    then ``x`` is an empty tensor holding ``u`` values at center such that
    ``x.shape[dim] == before + u.shape[dim] + after``. The empty values,
    original ``u`` values, and empty values are also arranged as such.

pad_beg : int
    The number of padding before the input values.

pad_end : int
    The number of padding after the input values.

dim : int
    The dimension to pad.

Returns
-------
x : torch.Tensor
    The same tensor after the padded values at ``dim`` are filled in.
"""

import torch


def _make_idx(item, dim, ndim):
    """
    Make an index that slices exactly along a specified dimension.

    Parameters
    ----------
    item : slice
        slice object at target axis

    dim : int
        target axis

    ndim : int
        total number of axes

    Returns
    -------
    tuple of slice
        Can be used to index np.ndarray and torch.Tensor
    """
    return (slice(None), ) * dim + (item, ) + (slice(None), ) * (ndim - dim - 1)


def symmetric_pad_1d(x, pad_beg, pad_end, dim):
    side_length = x.shape[dim]  # side length of padded tensor at this dimension

    def f(*args):  # helper function for _make_idx
        return _make_idx(slice(*args), dim=dim, ndim=x.ndim)

    while pad_beg > 0 or pad_end > 0:
        u_length = side_length - pad_beg - pad_end  # side length of "original" tensor
        if pad_beg > 0:  # symmetric pad at the beginning
            offset = min(pad_beg, u_length)
            x[f(pad_beg - offset, pad_beg)] = x[f(pad_beg, pad_beg + offset)].flip((dim,))
            pad_beg -= offset
        if pad_end > 0:  # symmetric pad at the end
            offset = min(pad_end, u_length)
            end_comp = side_length - pad_end  # end complement
            x[f(end_comp, end_comp + offset)] = x[f(end_comp - offset, end_comp)].flip((dim,))
            pad_end -= offset
    return x


def circular_pad_1d(x, pad_beg, pad_end, dim):
    side_length = x.shape[dim]  # side length of padded tensor at this dimension

    def f(*args):  # helper function for _make_idx
        return _make_idx(slice(*args), dim=dim, ndim=x.ndim)

    while pad_beg > 0 or pad_end > 0:
        u_length = side_length - pad_beg - pad_end  # side length of "original" tensor
        if pad_beg > 0:  # symmetric pad at the beginning
            offset = min(pad_beg, u_length)
            end_comp = side_length - pad_end
            x[f(pad_beg - offset, pad_beg)] = x[f(end_comp - offset, end_comp)]
            pad_beg -= offset
        if pad_end > 0:  # symmetric pad at the end
            offset = min(pad_end, u_length)
            end_comp = side_length - pad_end
            x[f(end_comp, end_comp + offset)] = x[f(pad_beg, pad_beg + offset)]
            pad_end -= offset
    return x


def reflect_pad_1d(x, pad_beg, pad_end, dim):
    side_length = x.shape[dim]  # side length of padded tensor at this dimension

    def f(*args):  # helper function for _make_idx
        return _make_idx(slice(*args), dim=dim, ndim=x.ndim)

    while pad_beg > 0 or pad_end > 0:
        u_length = side_length - pad_beg - pad_end  # side length of "original" tensor
        if pad_beg > 0:  # symmetric pad at the beginning
            offset = min(pad_beg, u_length - 1)
            x[f(pad_beg - offset, pad_beg)] = x[f(pad_beg + 1, pad_beg + 1 + offset)].flip((dim,))
            pad_beg -= offset
        if pad_end > 0:  # symmetric pad at the end
            offset = min(pad_end, u_length - 1)
            center = side_length - pad_end
            x[f(center, center + offset)] = x[f(center - 1 - offset, center - 1)].flip((dim,))
            pad_end -= offset
    return x



