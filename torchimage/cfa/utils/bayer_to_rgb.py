"""
Lightweight, deterministic methods to convert bayer images to RGB images.

Currently, only downsampling is implemented. A bayer image of shape ``(..., 1, 2h, 2w)`` or ``(..., 4, h, w)`` will be
converted to an RGB image of shape ``(..., 3, h, w)``. We condense every 2-by-2 block of bayer detector results into
a single pixel, taking R pixel's value as the new R, B pixel's value as the new B, and the average of 2 G pixel's values
as the new G.

For more advanced demosaicing algorithms (such as nearest neighbor, gradient-corrected bilinear interpolation, or
neural networks), please refer to other modules in ``demosaic``.
"""

import torch

from .bayer_1c_4c import bayer_1c_to_4c
from .validation import check_sensor_alignment, check_1c_bayer, check_4c_bayer


def downsample_1c_bayer_to_rgb(x: torch.Tensor, sensor_alignment: str):
    """
    Downsample a 1-channel bayer image to an RGB image.

    Input shape and sensor alignment are automatically validated, so it's not necessary to check again
    outside of the scope.

    Parameters
    ----------
    x : torch.Tensor
        Input 1-channel bayer array of shape ``(..., 1, 2h, 2w)``

    sensor_alignment : str
        Sensor alignment / bayer pattern

    Returns
    -------
    torch.Tensor
        Output 3-channel RGB array of shape ``(..., 3, h, w)``.

        Note that the output height and width are halved.
    """
    return downsample_4c_bayer_to_rgb(bayer_1c_to_4c(x), sensor_alignment=sensor_alignment)


def downsample_4c_bayer_to_rgb(x: torch.Tensor, sensor_alignment: str):
    """
    Downsample a 4-channel bayer image to an RGB image.

    Input shape and sensor alignment are automatically validated, so it's not necessary to check again
    outside of the scope.

    Parameters
    ----------
    x : torch.Tensor
        Input 4-channel bayer array of shape ``(..., 4, h, w)``

    sensor_alignment : str
        Sensor alignment / bayer pattern

    Returns
    -------
    torch.Tensor
        Output 3-channel RGB array of shape ``(..., 3, h, w)``.

    """
    sa = check_sensor_alignment(sensor_alignment)
    check_4c_bayer(x)
    ret_list = [
        x[..., sa.index("R"), :, :],
        (x[..., sa.index("G"), :, :] + x[..., sa.rindex("G"), :, :]) / 2,
        x[..., sa.index("B"), :, :]
    ]
    return torch.stack(ret_list, dim=-3)



