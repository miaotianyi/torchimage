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

negate : bool
    Whether to flip signs (+, -) when flipping the signal. Default: False.

    This parameter only applies to ``symmetric`` mode.
    When it is enabled, turns into half-sample antisymmetric mode:
        ``antisymmetric``: ``-d -c -b -a | a b c d | -d -c -b -a``

before, after : float
    For ``linear_ramp_1d``, they are the new edge values for the padded tensor. Default: 0

    For ``constant_1d``, they are the constants used for padding before and after ground truth.

    For ``stat_1d``, they are the lengths at the border to compute statistics with.

Returns
-------
x : torch.Tensor
    The same tensor after the padded values at ``dim`` are filled in.
"""

import torch

from .utils import make_idx, modify_idx


def replicate_1d(x, idx, dim):
    head, tail = idx[dim].start, idx[dim].stop

    def f(*args):  # fast idx modification
        return modify_idx(*args, idx=idx, dim=dim)

    if head > 0:  # should pad before
        x[f(head)] = x[f(head, head + 1)]

    if tail < x.shape[dim]:  # should pad after
        x[f(tail, None)] = x[f(tail - 1, tail)]

    return x


def circular_1d(x, idx, dim):
    head, tail = idx[dim].start, idx[dim].stop

    length = tail - head  # length of ground truth tensor at dim

    if length == 1:  # equivalent to replicate when there is only 1 point
        return replicate_1d(x, idx, dim)

    def f(*args):  # fast idx modification
        return modify_idx(*args, idx=idx, dim=dim)

    curr = head  # should pad before
    while curr > 0:
        chunk_size = min(curr, length)
        x[f(curr - chunk_size, curr)] = x[f(tail - chunk_size, tail)]
        curr -= chunk_size

    curr = tail  # should pad after
    while curr < x.shape[dim]:
        chunk_size = min(x.shape[dim] - curr, length)
        x[f(curr, curr + chunk_size)] = x[f(head, head + chunk_size)]
        curr += chunk_size

    return x


def periodize_1d(x, idx, dim):
    head, tail = idx[dim].start, idx[dim].stop

    def f(*args):  # fast idx modification
        return modify_idx(*args, idx=idx, dim=dim)

    length = tail - head  # length of ground truth tensor at dim

    if length % 2 == 0:
        return circular_1d(x, idx, dim)
    else:
        # replicate one element at last
        # note that in the outer scope, 1 extra padding space is always reserved
        # regardless of padding width
        # for example, even with padding width (0, 0),
        # (1, 2, 3) will still turn into (1, 2, 3, 3)
        new_idx = modify_idx(head, tail + 1, idx=idx, dim=dim)
        x[f(tail, tail + 1)] = x[f(tail - 1, tail)]  # replicate
        return circular_1d(x, new_idx, dim)


def symmetric_1d(x, idx, dim, negate=False):
    head, tail = idx[dim].start, idx[dim].stop
    length = tail - head  # length of ground truth tensor at dim

    if length == 1 and not negate:  # equivalent to replicate when there is only 1 point
        return replicate_1d(x, idx, dim)

    def f(*args):  # fast idx modification
        return modify_idx(*args, idx=idx, dim=dim)

    def g(*args):  # fast empty idx creation to index flipped cache
        return make_idx(*args, dim=dim, ndim=x.ndim)

    def h(a):  # conditionally flip the values of a tensor across 0
        return -a if negate else a

    if x.shape[dim] // length >= 2:
        # every column is flipped at least once
        # more advantageous to save as cache
        cache_flipped = h(x[idx].flip([dim]))
    else:
        cache_flipped = None

    curr = head  # should pad before
    flip = True  # whether to use flipped array for padding
    while curr > 0:
        chunk_size = min(curr, length)
        if flip:
            x[f(curr - chunk_size, curr)] = h(x[f(curr, curr + chunk_size)].flip([dim])) if cache_flipped is None else \
                                        cache_flipped[g(-chunk_size, None)]
        else:
            x[f(curr - chunk_size, curr)] = x[f(tail - chunk_size, tail)]
        curr -= chunk_size
        flip = not flip

    curr = tail  # should pad after
    flip = True
    while curr < x.shape[dim]:
        chunk_size = min(x.shape[dim] - curr, length)
        if flip:
            x[f(curr, curr + chunk_size)] = h(x[f(curr - chunk_size, curr)].flip([dim])) if cache_flipped is None else \
                                        cache_flipped[g(chunk_size)]
        else:
            x[f(curr, curr + chunk_size)] = x[f(head, head + chunk_size)]
        curr += chunk_size
        flip = not flip

    return x


def reflect_1d(x, idx, dim):
    head, tail = idx[dim].start, idx[dim].stop
    length = tail - head  # length of ground truth tensor at dim

    if length == 1:  # equivalent to replicate when there is only 1 point
        return replicate_1d(x, idx, dim)
    if length == 2:  # equivalent to circular when there are only 2 points
        return circular_1d(x, idx, dim)

    def f(*args):  # fast idx modification
        return modify_idx(*args, idx=idx, dim=dim)

    def g(*args):  # fast empty idx creation to index flipped cache
        return make_idx(*args, dim=dim, ndim=x.ndim)

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
        chunk_size = min(curr, length_flipped if flip else length)
        if flip:
            x[f(curr - chunk_size, curr)] = x[f(curr + 1, curr + 1 + chunk_size)].flip([dim]) if cache_flipped is None else \
                                        cache_flipped[g(-chunk_size, None)]
        else:
            x[f(curr - chunk_size, curr)] = x[f(tail - chunk_size, tail)]
        curr -= chunk_size
        flip = not flip

    curr = tail  # should pad after
    flip = True
    while curr < x.shape[dim]:
        chunk_size = min(x.shape[dim] - curr, length_flipped if flip else length)
        if flip:
            x[f(curr, curr + chunk_size)] = x[f(curr - 1 - chunk_size, curr - 1)].flip([dim]) if cache_flipped is None else \
                                        cache_flipped[g(chunk_size)]
        else:
            x[f(curr, curr + chunk_size)] = x[f(head, head + chunk_size)]
        curr += chunk_size
        flip = not flip
    return x


def odd_symmetric_1d(x, idx, dim):
    head, tail = idx[dim].start, idx[dim].stop

    def f(*args):  # fast idx modification
        return modify_idx(*args, idx=idx, dim=dim)

    # cache not implemented
    length = tail - head  # length of ground truth tensor at dim
    curr = head  # should pad before
    while curr > 0:
        chunk_size = min(curr, length)
        x[f(curr - chunk_size, curr)] = 2 * x[f(curr, curr + 1)] - x[f(curr, curr + chunk_size)].flip([dim])
        curr -= chunk_size

    curr = tail  # should pad after
    while curr < x.shape[dim]:
        chunk_size = min(x.shape[dim] - curr, length)
        x[f(curr, curr + chunk_size)] = 2 * x[f(curr - 1, curr)] - x[f(curr - chunk_size, curr)].flip([dim])
        curr += chunk_size
    return x


def odd_reflect_1d(x, idx, dim):
    head, tail = idx[dim].start, idx[dim].stop

    def f(*args):  # fast idx modification
        return modify_idx(*args, idx=idx, dim=dim)

    # cache not implemented
    length = tail - head  # length of ground truth tensor at dim
    length_flipped = length - 1  # reflect discards 1 border value

    curr = head  # should pad before
    while curr > 0:
        chunk_size = min(curr, length_flipped)
        x[f(curr - chunk_size, curr)] = 2 * x[f(curr, curr + 1)] - x[f(curr + 1, curr + 1 + chunk_size)].flip([dim])
        curr -= chunk_size

    curr = tail  # should pad after
    while curr < x.shape[dim]:
        chunk_size = min(x.shape[dim] - curr, length_flipped)
        x[f(curr, curr + chunk_size)] = 2 * x[f(curr - 1, curr)] - x[f(curr - 1 - chunk_size, curr - 1)].flip([dim])
        curr += chunk_size
    return x


def smooth_1d(x, idx, dim):
    head, tail = idx[dim].start, idx[dim].stop

    if tail - head == 1:
        return replicate_1d(x, idx=idx, dim=dim)

    def f(*args):  # fast idx modification
        return modify_idx(*args, idx=idx, dim=dim)

    def _arange(start, end, step):  # fast arange
        return torch.arange(start, end, step, dtype=x.dtype, device=x.device).view([-1] + [1] * (x.ndim - dim - 1))

    if head > 0:  # should pad before
        dist = _arange(head, 0, -1)  # distance from head
        x[f(head)] = x[f(head, head + 1)] - dist * (x[f(head + 1, head + 2)] - x[f(head, head + 1)])

    if tail < x.shape[dim]:  # should pad after
        dist = _arange(1, x.shape[dim] - tail + 1, 1)  # distance from tail
        x[f(tail, None)] = x[f(tail - 1, tail)] + dist * (x[f(tail - 1, tail)] - x[f(tail - 2, tail - 1)])

    return x


def linear_ramp_1d(x, idx, dim, before, after):
    head, tail = idx[dim].start, idx[dim].stop

    def f(*args):  # fast idx modification
        return modify_idx(*args, idx=idx, dim=dim)

    def _linspace(start, end, steps, s):  # fast linspace, s is slice object
        return torch.linspace(start, end, steps, dtype=x.dtype, device=x.device)[s].view([-1] + [1] * (x.ndim - dim - 1))

    if head > 0:  # should pad before
        dist = _linspace(0, 1, head + 1, slice(None, -1))  # distance from left end
        x[f(head)] = before + dist * (x[f(head, head + 1)] - before)

    if tail < x.shape[dim]:
        dist = _linspace(1, 0, x.shape[dim] - tail + 1, slice(1, None))  # distance from right end
        x[f(tail, None)] = after - dist * (after - x[f(tail - 1, tail)])

    return x


def constant_1d(x, idx, dim, before, after):
    head, tail = idx[dim].start, idx[dim].stop

    def f(*args):  # fast idx modification
        return modify_idx(*args, idx=idx, dim=dim)

    if head > 0:  # should pad before
        x[f(head)] = before

    if tail < x.shape[dim]:  # should pad after
        x[f(tail, None)] = after

    return x


_stat_funcs = {
    "mean": lambda x, dim: torch.mean(x, dim=dim, keepdim=True),
    "median": lambda x, dim: torch.median(x, dim=dim, keepdim=True).values,
    "maximum": lambda x, dim: torch.max(x, dim=dim, keepdim=True).values,
    "minimum": lambda x, dim: torch.min(x, dim=dim, keepdim=True).values
}


def stat_1d(x, idx, dim, before, after, mode):
    head, tail = idx[dim].start, idx[dim].stop
    length = tail - head

    def f(*args):  # fast idx modification
        return modify_idx(*args, idx=idx, dim=dim)

    sf = _stat_funcs[mode]  # statistic function

    if (head > 0 and tail < x.shape[dim]) and (before is None and after is None):
        # calculate total statistics only once
        result = sf(x[f(head, tail)], dim=dim)
        x[f(head)] = result
        x[f(tail, None)] = result

    if head > 0:  # should pad before
        if before is None:
            x[f(head)] = sf(x[f(head, tail)], dim=dim)
        else:
            before = min(length, max(1, before))  # stat length should be at least 1 & no greater than length
            x[f(head)] = sf(x[f(head, head + before)], dim=dim)

    if tail < x.shape[dim]:  # should pad after
        if after is None:
            x[f(tail, None)] = sf(x[f(head, tail)], dim=dim)
        else:
            after = min(length, max(1, after))  # stat length should be at least 1 & no greater than length
            x[f(tail, None)] = sf(x[f(tail - after, tail)], dim=dim)

    return x
