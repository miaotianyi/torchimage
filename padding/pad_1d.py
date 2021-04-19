"""
Pad a torch tensor along a certain dimension.

All 1-d padding utility functions share the same set of arguments.

**Important note**: This is an in-place function.

To avoid re-copying the input tensor, these 1d padding utility functions
only accept an empty tensor that has the same shape as the final padded output
and copies the values of the input tensor at its center.

The function will return its input x after modifying it.

**Efficiency Warning**: Currently, PyTorch doesn't support returning negative-strided
views like [::-1]. torch.flip() is said to be computationally expensive due to copying
the data, which might put ``symmetric`` and ``reflect`` to a disadvantage.

They are for lower-level utility only. Do *NOT* expect
them to behave the same way as ``F.pad`` does.

Comparing with ``torch.nn.functional.pad``, we make the following improvements:

1. Symmetric padding is added

2. Higher-dimension non-constant padding
    To the date of this release, PyTorch has not implemented reflect (ndim >= 3+2),
    replicate (ndim >= 4+2), and circular (ndim >= 4+2) for high-dimensional
    tensors (the +2 refers to the initial batch and channel dimensions).

    This is achieved by decomposing n-dimensional padding to sequential padding
    along every dimension of interest.

3. Wider padding size
    Padding modes reflect and circular will cause PyTorch to fail when padding size
    is greater than the tensor's shape at a certain dimension.
    i.e. Padding value causes wrapping around more than once.

We design this padding package with these principles:

1. Maximal compatibility with PyTorch.
    We will implicitly use ``F.pad`` as much as we can.
    (For example, we do not implement zero and constant padding)
    We also adjust the names of the arguments to be compatible with PyTorch

2. Versatility
    We try to incorporate as many functionalities available in other packages
    as possible. Specifically, we try to mimic the behavior of ``numpy.pad``


Naming Convention
------------
The naming convention of padding modes is contested.
- Scipy's ``mirror`` is PyTorch's ``reflect``.
- Scipy's ``wrap`` is PyTorch's ``circular``.
- Scipy's ``reflect`` is implemented as ``symmetric`` here;
    it has no counterpart in PyTorch.

``"empty"`` - pads with undefined values
    ``? ? ? ? | a b c d | ? ? ? ?`` where the empty values are from ``torch.empty``

``"constant"`` - pads with a constant value
    ``p p p p | a b c d | p p p p`` where ``p`` is supplied by ``padding_value`` argument.

``"zeros"`` - pads with 0; a special case of constant padding
    ``0 0 0 0 | a b c d | 0 0 0 0``

``"symmetric"`` - extends signal by *mirroring* samples. Also known as half-sample symmetric
    ``d c b a | a b c d | d c b a``

``"antisymmetric"`` - extends signal by *mirroring* and *negating* samples. Also known as half-sample antisymmetric.
    ``-d -c -b -a | a b c d | -d -c -b -a``

``"reflect"`` - signal is extended by *reflecting* samples. This mode is also known as whole-sample symmetric
    ``d c b | a b c d | c b a``

``"replicate"`` - replicates the border pixels
    ``a a a a | a b c d | d d d d``

``"circular"`` - signal is treated as a periodic one
    ``a b c d | a b c d | a b c d``

========================       =============   ===========     ==============================  =======
torchimage                     PyWavelets      Matlab          numpy.pad                       Scipy
========================       =============   ===========     ==============================  =======
symmetric                      symmetric       sym, symh       symmetric                       reflect
reflect                        reflect         symw            reflect                         mirror
todo                           smooth          spd, sp1        N/A                             N/A
replicate                      constant        sp0             edge                            nearest
zeros                          zero            zpd             constant, cval=0                N/A
constant                       N/A             N/A             constant                        constant
circular                       periodic        ppd             wrap                            wrap
todo                           periodization   per             N/A                             N/A
symmetric, negative=True       antisymmetric   asym, asymh     N/A                             N/A
todo                           antireflect     asymw           reflect, reflect_type='odd'     N/A
todo                           N/A             N/A             symmetric, reflect_type='odd'   N/A
?                                                              linear_ramp                     N/A
?                                                              maximum, mean, median, minimum  N/A
empty                          N/A             N/A             empty                           N/A
                                                               <function>                      N/A
========================       =============   ===========     ==============================  =======


Parameters
----------
x : torch.Tensor
    The input tensor to be padded.

    See the important note above. Let ``u`` be the original tensor,
    then ``x`` is an empty tensor holding ``u`` values at center such that
    ``x[idx] == u``

idx : tuple of slice
    Indices for the ground truth tensor located at the center of the
    empty-padded tensor x.

    Has the same length as the number of dimensions ``len(idx) == x.ndim``.
    Each element is a ``slice(beg, end, 1)`` where at dimension ``dim``,
    ``x.shape[dim] - end`` is the amount of padding in the end and
    ``beg`` is the amount of padding in the beginning.

    Note that this has to be a tuple to properly index a high-dimensional
    tensor.

    This tuple of index slices prevents computing padding for empty values
    at this dimension.

dim : int
    The dimension to pad.

Other Parameters
----------------
These keyword arguments are used in some padding functions only.

negative : bool
    Whether to flip signs (+, -) when flipping the signal. Default: False.

    This parameter only applies to ``symmetric`` mode.
    When it is enabled, turns into half-sample antisymmetric mode:
        ``antisymmetric``: ``-d -c -b -a | a b c d | -d -c -b -a``

Returns
-------
x : torch.Tensor
    The same tensor after the padded values at ``dim`` are filled in.
"""

import torch


def _make_idx(*args, dim, ndim):
    """
    Make an index that slices exactly along a specified dimension.
    e.g. [:, ... :, slice(*args), :, ..., :]

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
    idx : tuple of slice
        Can be used to index np.ndarray and torch.Tensor
    """
    return (slice(None), ) * dim + (slice(*args), ) + (slice(None), ) * (ndim - dim - 1)


def _modify_idx(*args, idx, dim):
    new_idx = list(idx)
    new_idx[dim] = slice(*args)
    return tuple(new_idx)


def replicate_1d(x, idx, dim):
    head, tail = idx[dim].start, idx[dim].stop

    def f(*args):  # fast idx modification
        return _modify_idx(*args, idx=idx, dim=dim)

    if head > 0:  # should pad before
        x[f(head)] = x[f(head, head + 1)]

    if tail < x.shape[dim]:  # should pad after
        x[f(tail, None)] = x[f(tail - 1, tail)]

    return x


def circular_1d(x, idx, dim):
    head, tail = idx[dim].start, idx[dim].stop

    def f(*args):  # fast idx modification
        return _modify_idx(*args, idx=idx, dim=dim)

    length = tail - head  # length of ground truth tensor at dim

    curr = head  # should pad before
    while curr > 0:
        offset = min(curr, length)
        x[f(curr - offset, curr)] = x[f(tail - offset, tail)]
        curr -= offset

    curr = tail  # should pad after
    while curr < x.shape[dim]:
        offset = min(x.shape[dim] - curr, length)
        x[f(curr, curr + offset)] = x[f(head, head + offset)]
        curr += offset

    return x


def symmetric_1d(x, idx, dim, negative=False):
    head, tail = idx[dim].start, idx[dim].stop

    def f(*args):  # fast idx modification
        return _modify_idx(*args, idx=idx, dim=dim)

    def g(*args):  # fast empty idx creation to index flipped cache
        return _make_idx(*args, dim=dim, ndim=x.ndim)

    def h(a):  # conditionally flip the values of a tensor across 0
        return -a if negative else a

    length = tail - head  # length of ground truth tensor at dim

    if x.shape[dim] // length >= 2:
        # every column is flipped at least once
        # more advantageous to save as cache
        cache_flipped = h(x[idx].flip([dim]))
    else:
        cache_flipped = None

    curr = head  # should pad before
    flip = True  # whether to use flipped array for padding
    while curr > 0:
        offset = min(curr, length)
        if flip:
            x[f(curr - offset, curr)] = h(x[f(curr, curr + offset)].flip([dim])) if cache_flipped is None else \
                                        cache_flipped[g(-offset, None)]
        else:
            x[f(curr - offset, curr)] = x[f(tail - offset, tail)]
        curr -= offset
        flip = not flip

    curr = tail  # should pad after
    flip = True
    while curr < x.shape[dim]:
        offset = min(x.shape[dim] - curr, length)
        if flip:
            x[f(curr, curr + offset)] = h(x[f(curr - offset, curr)].flip([dim])) if cache_flipped is None else \
                                        cache_flipped[g(offset)]
        else:
            x[f(curr, curr + offset)] = x[f(head, head + offset)]
        curr += offset
        flip = not flip

    return x


def reflect_1d(x, idx, dim):
    head, tail = idx[dim].start, idx[dim].stop

    def f(*args):  # fast idx modification
        return _modify_idx(*args, idx=idx, dim=dim)

    def g(*args):  # fast empty idx creation to index flipped cache
        return _make_idx(*args, dim=dim, ndim=x.ndim)

    length = tail - head  # length of ground truth tensor at dim
    length_flipped = length - 2  # reflect discards 2 border values

    if x.shape[dim] // length >= 2:
        # every column is flipped at least once
        # more advantageous to save as cache
        cache_flipped = x[f(head + 1, tail - 1)].flip([dim])
    else:
        cache_flipped = None

    curr = head  # should pad before
    flip = True  # whether to use flipped array for padding
    while curr > 0:
        offset = min(curr, length_flipped if flip else length)
        if flip:
            x[f(curr - offset, curr)] = x[f(curr + 1, curr + 1 + offset)].flip([dim]) if cache_flipped is None else \
                                        cache_flipped[g(-offset, None)]
        else:
            x[f(curr - offset, curr)] = x[f(tail - offset, tail)]
        curr -= offset
        flip = not flip

    curr = tail  # should pad after
    flip = True
    while curr < x.shape[dim]:
        offset = min(x.shape[dim] - curr, length_flipped if flip else length)
        if flip:
            x[f(curr, curr + offset)] = x[f(curr - 1 - offset, curr - 1)].flip([dim]) if cache_flipped is None else \
                                        cache_flipped[g(offset)]
        else:
            x[f(curr, curr + offset)] = x[f(head, head + offset)]
        curr += offset
        flip = not flip

    return x


def odd_symmetric_1d(x, idx, dim):
    pass


def odd_reflect_1d(x, idx, dim):
    pass


def smooth_1d(x, idx, dim):
    head, tail = idx[dim].start, idx[dim].stop

    def f(*args):  # fast idx modification
        return _modify_idx(*args, idx=idx, dim=dim)

    if head > 0:  # should pad before
        dist = torch.arange(head, 0, -1).view([-1] + [1] * (x.ndim - dim - 1))
        x[f(head)] = x[f(head, head + 1)] - dist * (x[f(head + 1, head + 2)] - x[f(head, head + 1)])

    if tail < x.shape[dim]:  # should pad after
        dist = torch.arange(1, x.shape[dim] - tail + 1).view([-1] + [1] * (x.ndim - dim - 1))
        x[f(tail, None)] = x[f(tail - 1, tail)] + dist * (x[f(tail - 1, tail)] - x[f(tail - 2, tail - 1)])

    return x
