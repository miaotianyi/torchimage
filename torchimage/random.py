import torch

from .utils import NdSpec
from .utils.validation import check_axes


def random_crop(x: torch.Tensor, axes, size, *, generator: torch.Generator = None):
    axes = check_axes(x, axes)
    size = NdSpec(size, item_shape=[])
    idx = [slice(None)] * x.ndim
    for i, a in enumerate(axes):
        length = size[i]
        if length is None:
            continue
        beg = torch.randint(x.shape[a] - length, size=(), generator=generator).item()
        end = beg + length
        idx[a] = slice(beg, end)
    return x[idx]


def add_poisson_gauss_noise(x: torch.Tensor, k, sigma, *, generator: torch.Generator = None) -> torch.Tensor:
    """
    Add Poisson-Gaussian noise to a tensor.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    k : float
        Noise parameter for poisson

    sigma_2 : float
        Noise parameter for gaussian distribution (standard deviation)

    generator : torch.Generator
        Torch random number generator

    Returns
    -------
    y : torch.Tensor
        Output tensor with noise added
    """
    sigma_2 = sigma ** 2

    poisson_noise = torch.poisson(x / k, generator=generator)
    poisson_noise *= k  # inplace to save memory
    gaussian_noise = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)
    gaussian_noise *= (sigma_2 ** 0.5)  # inplace to save memory
    y = poisson_noise
    y += gaussian_noise
    return y


def add_gauss_noise(x: torch.Tensor, sigma, *, generator: torch.Generator = None) -> torch.Tensor:
    sigma_2 = sigma ** 2
    gaussian_noise = torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)
    gaussian_noise *= (sigma_2 ** 0.5)  # inplace to save memory
    gaussian_noise += x  # inplace
    return gaussian_noise
