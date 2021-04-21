import torch
from torch.nn import functional as F

from . import pad_1d
from .utils import _modify_idx, _check_padding


_padding_function_dict = {
    "replicate": pad_1d.replicate_1d,
    "smooth": pad_1d.smooth_1d,
    "circular": pad_1d.circular_1d,
    "periodize": pad_1d.periodize_1d,
    "symmetric": lambda x, idx, dim: pad_1d.symmetric_1d(x, idx, dim, negate=False),
    "reflect": pad_1d.reflect_1d,
    "antisymmetric": lambda x, idx, dim: pad_1d.symmetric_1d(x, idx, dim, negate=True),
    "odd_reflect": pad_1d.odd_reflect_1d,
    "odd_symmetric": pad_1d.odd_symmetric_1d,
}


def pad(x: torch.Tensor, pad, mode, constant_value=0):
    """
    Pad an n-dimensional tensor according to a user-specified mode.

    Parameters
    ----------
    x : torch.Tensor
        Original tensor to be padded

    pad : tuple of int
        Padding width specification.

        It must be a tuple like
        ``(before_{n-1}, after_{n-1}, before_{n-2}, after_{n-2}, ..., before_{dim}, after_{dim})``,
        such that dimensions before ``dim`` is not padded.

        This is consistent with PyTorch's padding width specification.
        However, unlike ``np.pad``, this argument does *not* allow broadcasting
        ``(before, after)`` to all dimensions, since this would have been
        confused with ``(before_{n-1}, after_{n-1})``.

    mode : str or <function>
        Padding mode or padding function.

    constant_value : float, tuple of float
        Constant value for padding if mode is ``constant``.

        If it's a tuple of float, it must follow the format of torch padding width.
        e.g. ``(before_{n-1}, after_{n-1}, before_{n-2}, after_{n-2}, ..., before_{dim}, after_{dim})``

    Returns
    -------
    y : torch.Tensor
        New padded tensor
    """
    # fall back to torch's official implementation
    if mode == "zeros":
        return F.pad(x, pad=pad, mode="constant", value=0)
    if mode == "constant":
        return F.pad(x, pad=pad, mode=mode, value=constant_value)

    # input validation
    _check_padding(x, pad)

    ndim_padded = len(pad) // 2  # the number of dimensions that are padded

    pad_beg = torch.tensor([0] * (x.ndim - ndim_padded) + list(pad[::2][::-1]))
    pad_end = torch.tensor([0] * (x.ndim - ndim_padded) + list(pad[1::2][::-1]))

    old_shape = torch.tensor(x.shape)

    head = pad_beg
    tail = head + old_shape
    new_shape = tail + pad_end

    if mode == "periodize":
        new_shape += old_shape % 2  # if any dimension is odd, automatically add one padding

    if (old_shape == new_shape).all():
        return x  # no padding needed, avoid creating new empty tensor

    y = torch.empty(tuple(new_shape.tolist()), dtype=x.dtype, device=x.device)
    # copy original elements
    idx = tuple([slice(start, stop) for start, stop in zip(head.tolist(), tail.tolist())])
    y[idx] = x

    if mode == "empty":
        return y

    if callable(mode):  # cusotmized padding function
        pad_func = mode
    elif mode not in _padding_function_dict:
        raise ValueError(f"Unsupported padding mode {mode}")
    else:
        pad_func = _padding_function_dict[mode]

    for dim in range(x.ndim):
        if head[dim] == 0 and tail[dim] == y.shape[dim]:  # no padding required
            continue
        y = pad_func(y, idx, dim)  # note that pad_func is still in-place
        idx = _modify_idx(None, idx=idx, dim=dim)

    return y
