import torch
from torch import nn
from torch.nn import functional as F

from . import pad_1d
from .utils import _modify_idx, _check_padding, pad_width_format


class GenericPadNd(nn.Module):
    def __init__(self, pad_width, mode, constant_values, end_values, stat_length):
        super(GenericPadNd, self).__init__()

    def forward(self, x: torch.Tensor):  # all at once; pad all specified dimensions
        pass

    def pad_dim(self, x: torch.Tensor, dim: int):  # pad a specific dimension
        pass

