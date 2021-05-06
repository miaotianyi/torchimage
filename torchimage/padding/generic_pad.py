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

        # number of padded dimensions
        # broadcast if ndim == 0, always right-justify
        # crop self left if self.ndim > x.ndim
        # ignore x left if self.ndim < x.ndim

        # list of NdSpec lengths
        ndim_list = list(map(len, [
            self.pad_width, self.mode, self.constant_values, self.end_values, self.stat_length]))
        # unique NdSpec lengths that are non-broadcastable (length > 0)
        pos_ndim_set = set(x for x in ndim_list if x > 0)


    def forward(self, x: torch.Tensor, axes=None):  # all at once; pad all specified dimensions
        # if axes are None, automatic right-justify, no error should be raised
        # otherwise, it should be a slice or iterable
        # assert len(self.pad_width) <= x.ndim
        pass

    def pad_axis(self, x: torch.Tensor, axis: int):  # pad a specific dimension
        pass

