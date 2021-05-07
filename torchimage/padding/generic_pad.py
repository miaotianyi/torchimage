import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from torchimage.utils import NdSpec
from . import pad_1d
from .utils import modify_idx, _check_padding, pad_width_format

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
    def __init__(self, pad_width, mode, constant_values=0, end_values=0.0, stat_length=None):
        super(GenericPadNd, self).__init__()
        self.pad_width = NdSpec(pad_width, item_shape=[2])
        assert all(pw[0] >= 0 <= pw[1] for pw in self.pad_width)
        self.mode = NdSpec(mode)
        assert all(_check_padding_mode(m) for m in self.mode)
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

        if axes is None:
            axes = tuple(range(x.ndim))
        elif isinstance(axes, slice):
            axes = tuple(range(x.ndim)[axes])
        else:
            axes = tuple(int(a) for a in axes)
            assert all(0 <= a <= x.ndim for a in axes)
            assert len(set(axes)) == len(axes)  # no repeated axes

        # reverse mapping
        axes_to_ind = {axes[i]: i for i in range(-len(axes), 0)}

        ndim_padded = min(x.ndim, self.ndim, len(axes))

        old_shape_vec = np.array(x.shape)
        head_vec = np.zeros(x.ndim)  # equal to pad_before
        pad_after_vec = np.zeros(x.ndim)

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
        idx = tuple([slice(start, stop) for start, stop in zip(head_vec.tolist(), tail_vec.tolist())])
        y[idx] = x

        for i in range(-ndim_padded, 0):
            pw = self.pad_width[i]
            if pw[0] == pw[1] == 0:  # no padding required
                continue

            mode = self.mode[i]
            if mode == "empty":
                continue

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

            a = axes[i]  # axis (dimension) in x
            y = pad_func(y, idx, a)  # note that pad_func is still in-place

            new_head = idx[a].start - pw[0]
            new_tail = idx[a].stop + pw[1] + ((idx[a].stop - idx[a].start) % 2 if mode == "periodize" else 0)
            idx = modify_idx(new_head, new_tail, idx=idx, dim=a)

        return y

    def pad_axis(self, x: torch.Tensor, axis: int):  # pad a specific axis
        pass

