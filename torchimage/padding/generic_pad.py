import numpy as np
import torch
from torch import nn

from torchimage.utils import NdSpec
from . import pad_1d
from .utils import modify_idx, make_idx
from ..utils.validation import check_axes

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

_other_padding_set = {
    "zeros", "constant", "linear_ramp", "empty"
}


def _check_padding_mode(mode):
    return callable(mode) or mode in set().union(_stat_padding_set)\
        .union(_other_padding_set).union(_padding_function_dict.keys())


class GenericPadNd(nn.Module):
    def __init__(self, pad_width=0, mode="constant", constant_values=0, end_values=0.0, stat_length=None):
        """
        Parameters
        ----------
        pad_width : int, pair of int, list of pair of int
            Padding width specification.

            If it's a single integer ``width``, all axes of interest
            (specified by the ``axes`` parameter in forward function)
            will have ``width`` padded entries before and after the ground
            truth region.

            If it's a pair of integers ``(before, after)``, then every
            axis of interest will have ``before`` padded entries before
            and ``after`` padded entries after the ground truth region.

            If it's a list of pairs of integers, it's interpreted as a
            right-justified, left-to-right, and nested NdSpec format:
            ``[..., (before_{-2}, after_{-2}), (before_{-1}, after_{-1})]``

        mode : str, <function>, list of str or <function>
            Padding mode or padding function.

        constant_values : float, pair of float, list of pair of float
            Constant value for padding if mode is ``constant``.

        end_values : float, pair of float, list of pair of float
            Used in ``linear_ramp``. The values used for the ending value of the linear ramp
            and that will form the edge of the padded array. Default: 0

        stat_length : None, int, pair of int, list of pair of int
            Used in "maximum", "mean", "median", and "minimum".
            Number of values at edge of each axis used to calculate the statistic value.
            Default: None.

            If None, all values at the axis will be used for calculation.
            (This is computationally expensive, not recommended.)

            For each axis, this number will be clipped by ``(1, length)`` where
            ``length`` is the side length of the original tensor.
        """
        super(GenericPadNd, self).__init__()

        self.pad_width = NdSpec(pad_width, item_shape=[2])

        def _check_pad_width(pw):
            assert pw[0] >= 0 <= pw[1]

        self.pad_width.map(_check_pad_width)

        self.mode = NdSpec(mode)
        self.mode.map(_check_padding_mode)

        self.constant_values = NdSpec(constant_values, item_shape=[2])
        self.end_values = NdSpec(end_values, item_shape=[2])
        self.stat_length = NdSpec(stat_length, item_shape=[2])

        # list of NdSpec lengths
        ndim_list = list(map(len, [
            self.pad_width, self.mode, self.constant_values, self.end_values, self.stat_length]))
        # unique NdSpec lengths that are non-broadcastable (length > 0)
        pos_ndim_set = set(x for x in ndim_list if x > 0)
        if len(pos_ndim_set) > 1:
            raise ValueError(
                f"Incompatible non-broadcastable NdSpec lengths: " +
                (f"{self.pad_width=} " if len(self.pad_width) > 0 else "") +
                (f"{self.mode=} " if len(self.mode) > 0 else "") +
                (f"{self.constant_values=} " if len(self.constant_values) > 0 else "") +
                (f"{self.end_values=} " if len(self.end_values) > 0 else "") +
                (f"{self.stat_length=} " if len(self.stat_length) > 0 else "")
            )

        if pos_ndim_set:  # must have size 1
            self.ndim = next(iter(pos_ndim_set))  # fixed-length; non-broadcastable
        else:
            self.ndim = 0  # broadcastable

    def _fill_value_1d(self, y: torch.Tensor, i: int, axis: int, idx):
        """
        Fill padding values in place using self (padder) specification
        at ``i`` along y (tensor) dimension ``axis``.

        Parameters
        ----------
        y : torch.Tensor
            An empty tensor enclosing a ground truth region ``y[idx]``

        i : int
            Axis for padder (may be different from axis); can be positive or negative

        axis : int
            Axis for tensor (may be different from axis); can be positive or negative

        idx : tuple of slice
            Slice for the region of ground truth.

        Returns
        -------
        idx : tuple of slice
            Modified idx that now marks the recently padded regions
            as ground truth as well.
        """
        pw = self.pad_width[i]
        if pw[0] == pw[1] == 0:  # no padding required
            return idx

        mode = self.mode[i]
        if mode == "empty":
            return idx

        # determine pad_func at each iteration
        if callable(mode):
            pad_func = mode
        elif mode == "constant":
            cv = self.constant_values[i]

            def pad_func(x, idx, dim):
                return pad_1d.constant_1d(x, idx, dim, before=cv[0], after=cv[1])
        elif mode == "linear_ramp":  # linear ramp
            ev = self.end_values[i]

            def pad_func(x, idx, dim):
                return pad_1d.linear_ramp_1d(x, idx, dim, before=ev[0], after=ev[1])
        elif mode in _stat_padding_set:
            sl = self.stat_length[i]

            def pad_func(x, idx, dim):
                return pad_1d.stat_1d(x, idx, dim, before=sl[0], after=sl[1], mode=mode)
        elif mode not in _padding_function_dict:
            raise ValueError(f"Unsupported padding mode {mode}")
        else:  # no other keyword arguments required
            pad_func = _padding_function_dict[mode]

        # axis: dimension in x
        pad_func(y, idx=idx, dim=axis)  # note that pad_func is still in-place

        new_head = idx[axis].start - pw[0]
        new_tail = idx[axis].stop + pw[1]
        if mode == "periodize":
            new_tail += (idx[axis].stop - idx[axis].start) % 2
        idx = modify_idx(new_head, new_tail, idx=idx, dim=axis)
        return idx

    def forward(self, x: torch.Tensor, axes=None):
        """
        Pads a tensor sequentially at all specified dimensions

        Parameters
        ----------
        x : torch.Tensor
            The input n-dimensional tensor to be padded.

        axes : sequence of int, slice, None
            The sequence of dimensions to be padded with the exact ordering.

            If axes is not provided (None), the padder will automatically right-justify
            the NdSpecs such that the "rightmost" axis corresponds to the "rightmost"
            item in an NdSpec and other entries are aligned accordingly.

            If the padder has larger ndim than x (or axes), the leftmost dimensions
            in the padder are ignored. If x (or axes) has larger ndim than the padder,
            the leftmost dimensions of x are left unchanged.

            If the input format is PyTorch batch-like (first 2 dimensions are batch
            and channel dimensions), we recommend using ``axes=slice(2, None)``.

        Returns
        -------
        y : torch.Tensor
            Padded tensor.
        """
        assert x.ndim > 0  # must have at least 1 dimension

        axes = check_axes(x, axes)

        ndim_padded = min(x.ndim, len(axes))
        if self.ndim > 0:
            ndim_padded = min(ndim_padded, self.ndim)

        old_shape_vec = np.array(x.shape, dtype=int)
        head_vec = np.zeros(x.ndim, dtype=int)  # equal to pad_before
        pad_after_vec = np.zeros(x.ndim, dtype=int)

        for i in range(-ndim_padded, 0):
            pw = self.pad_width[i]
            a = axes[i]
            head_vec[a] += pw[0]
            pad_after_vec[a] += pw[1]
            if self.mode[i] == "periodize":
                pad_after_vec[a] += old_shape_vec[a] % 2

        tail_vec = head_vec + old_shape_vec
        new_shape_vec = tail_vec + pad_after_vec

        if (old_shape_vec == new_shape_vec).all():
            return x  # no padding needed, avoid creating new empty tensor

        y = torch.empty(tuple(new_shape_vec.tolist()), dtype=x.dtype, device=x.device)
        # copy original elements
        idx = tuple([slice(start, stop) for start, stop in zip(head_vec, tail_vec)])
        y[idx] = x

        for i in range(-ndim_padded, 0):
            idx = self._fill_value_1d(y, i=i, axis=axes[i], idx=idx)

        return y

    def pad_axis(self, x: torch.Tensor, axis: int):  # pad a specific axis
        """
        Pad a specific axis according to predefined padder parameters

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to be padded.

        axis : int
            The axis in x to be padded. Can be a positive or negative index.

        Returns
        -------
        y : torch.Tensor
            Padded tensor.
        """
        assert -x.ndim <= axis < x.ndim  # is valid index
        axis = axis if axis < 0 else axis - x.ndim  # right-justify with negative index

        pw = self.pad_width[axis]
        if pw[0] == pw[1] == 0:  # no padding required
            return x

        head = pw[0]
        tail = head + x.shape[axis]
        length = tail + pw[1]

        idx = make_idx(head, tail, dim=axis, ndim=x.ndim)

        new_shape_vec = list(x.shape)
        new_shape_vec[axis] = length

        y = torch.empty(tuple(new_shape_vec), dtype=x.dtype, device=x.device)
        y[idx] = x
        self._fill_value_1d(y, i=axis, axis=axis, idx=idx)
        return y

