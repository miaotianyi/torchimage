import torch
from torch.nn import functional as F

from . import pad_1d
from .utils import _modify_idx, _check_padding, pad_width_format


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


_stat_padding_set = {
    "maximum", "mean", "median", "minimum"
}


def pad(x: torch.Tensor, padding, mode, constant_values=0, end_values=0.0, stat_length=None):
    """
    Pad an n-dimensional tensor according to a user-specified mode.

    Parameters
    ----------
    x : torch.Tensor
        Original tensor to be padded

    padding : tuple of int
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

    constant_values : float, tuple of float
        Constant value for padding if mode is ``constant``.

        If it's a tuple of float, it must follow the format of torch padding width.
        e.g. ``(before_{n-1}, after_{n-1}, before_{n-2}, after_{n-2}, ..., before_{dim}, after_{dim})``

    end_values : float, tuple of float
        Used in ``linear_ramp``. The values used for the ending value of the linear_ramp
        and that will form the edge of the padded array. Default: 0

        If it's a tuple of float, it must follow the format of torch padding width.
        e.g. ``(before_{n-1}, after_{n-1}, before_{n-2}, after_{n-2}, ..., before_{dim}, after_{dim})``

    stat_length : None, int, tuple of int
        Used in "maximum", "mean", "median", and "minimum".
        Number of values at edge of each axis used to calculate the statistic value.
        Default: None.

        If None, all values at the axis will be used for calculation.
        (This is computationally expensive, not recommended.)

        For each axis, this number will be clipped by ``(1, length)`` where
        ``length`` is the side length of the original tensor.

    Returns
    -------
    y : torch.Tensor
        New padded tensor
    """
    # fall back to torch's official implementation
    if mode == "zeros":
        return F.pad(x, pad=padding, mode="constant", value=0)
    if mode == "constant" and not hasattr(constant_values, "__len__"):  # 1 constant value for padding
        return F.pad(x, pad=padding, mode=mode, value=constant_values)

    # input validation
    _check_padding(x, padding)

    pad_beg, pad_end = torch.tensor(pad_width_format(padding, source="torch", target="numpy", ndim=x.ndim)).T
    # ndim_padded = len(pad) // 2  # the number of dimensions that are padded
    # pad_beg = torch.tensor([0] * (x.ndim - ndim_padded) + list(pad[::2][::-1]))
    # pad_end = torch.tensor([0] * (x.ndim - ndim_padded) + list(pad[1::2][::-1]))

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

    if callable(mode):  # customized padding function
        pad_func = mode
    elif mode == "constant":  # constant padding with multiple values
        assert len(constant_values) == len(padding)
        values = pad_width_format(constant_values, source="torch", target="numpy", ndim=x.ndim)

        def pad_func(x, idx, dim):
            return pad_1d.constant_1d(x, idx, dim, before=values[dim][0], after=values[dim][1])
    elif mode == "linear_ramp":  # linear ramp
        if not hasattr(end_values, "__len__"):  # end_values is scalar
            values = ((end_values, end_values),) * x.ndim  # same end values everywhere
        else:  # end_values is tuple with torch padding width format
            values = pad_width_format(end_values, source="torch", target="numpy", ndim=x.ndim)

        def pad_func(x, idx, dim):
            return pad_1d.linear_ramp_1d(x, idx, dim, before=values[dim][0], after=values[dim][1])
    elif mode in ("maximum", "mean", "median", "minimum"):
        if not hasattr(stat_length, "__len__"):  # end_values is scalar
            values = ((stat_length, stat_length),) * x.ndim  # same end values everywhere
        else:  # end_values is tuple with torch padding width format
            values = pad_width_format(stat_length, source="torch", target="numpy", ndim=x.ndim)

        def pad_func(x, idx, dim):
            return pad_1d.stat_1d(x, idx, dim, before=values[dim][0], after=values[dim][1], mode=mode)
    elif mode not in _padding_function_dict:
        raise ValueError(f"Unsupported padding mode {mode}")
    else:  # no other keyword arguments required
        pad_func = _padding_function_dict[mode]

    for dim in range(x.ndim):
        if head[dim] == 0 and tail[dim] == y.shape[dim]:  # no padding required
            continue
        y = pad_func(y, idx, dim)  # note that pad_func is still in-place
        idx = _modify_idx(None, idx=idx, dim=dim)

    return y
