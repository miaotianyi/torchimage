import torch

from .utils import NdSpec
from .utils.validation import check_axes


def random_crop(x: torch.Tensor, axes, sizes, generator: torch.Generator = None):
    axes = check_axes(x, axes)
    sizes = NdSpec(sizes, item_shape=[])
    idx = [slice(None)] * x.ndim
    for i, a in enumerate(axes):
        length = sizes[i]
        beg = torch.randint(x.shape[a] - length, size=(), generator=generator).item()
        end = beg + length
        idx[a] = slice(beg, end)
    return x[idx]

