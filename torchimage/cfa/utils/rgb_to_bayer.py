"""
Convert RGB images to bayer images by adding mosaic.

For every pixel, 2 out of the 3 colors will be discarded while only 1 remains. Which color remains depends
on the sensor alignment specified by the parameters. Therefore, this process is:

1. Irreversible: Since converting RGB to bayer involves a loss of information,
its inverse (demosaicing) is a much harder problem.
2. Deterministic

This module works with generalized nchw format.

We separate converting to 1-channel bayer and 4-channel bayer into 2 functions.
This is because conversion from RGB to bayer, as well as conversion between different bayer formats,
all requires creating new tensors in memory. 2-step conversion (e.g. RGB to 1c bayer, then to 4c bayer)
will bear the overhead from assigning and releasing memory for intermediate tensors.

See Also
--------
MatLab demosaic: https://de.mathworks.com/help/images/ref/demosaic.html
"""

import torch

from .validation import check_sensor_alignment, check_rgb


_color_dict = {
    "R": 0,
    "G": 1,
    "B": 2
}


def rgb_to_1c_bayer(x: torch.Tensor, sensor_alignment: str):
    """
    Convert RGB image to 1-channel bayer image.

    Input shape and sensor alignment are automatically validated, so it's not necessary to check again
    outside of the scope.

    Parameters
    ----------
    x : torch.Tensor
        Input RGB image of shape ``(..., 3, h, w)``, where ``h`` and ``w`` must be even
    sensor_alignment : str
        Sensor alignment / bayer pattern

    Returns
    -------
    y : torch.Tensor
        Output 1-channel bayer image of shape ``(..., 1, h, w)``.

        Has the same dtype and device as input.
    """
    check_rgb(x)
    sensor_alignment = check_sensor_alignment(sensor_alignment)
    h, w = x.shape[-2], x.shape[-1]
    assert h % 2 == w % 2 == 0  # input rgb image must have even-length height and width
    y = torch.empty(x.shape[:-3] + (1, h, w), dtype=x.dtype, device=x.device)
    for r in 0, 1:
        for c in 0, 1:
            color_ind = _color_dict[sensor_alignment[r * 2 + c]]
            y[..., 0, r::2, c::2] = x[..., color_ind, r::2, c::2]
    return y


def rgb_to_4c_bayer(x: torch.Tensor, sensor_alignment: str):
    """
    Convert RGB image to 4-channel bayer image.

    Input shape and sensor alignment are automatically validated, so it's not necessary to check again
    outside of the scope.

    Parameters
    ----------
    x : torch.Tensor
        Input RGB image of shape ``(..., 3, h, w)``, where ``h`` and ``w`` must be even
    sensor_alignment : str
        Sensor alignment / bayer pattern

    Returns
    -------
    y : torch.Tensor
        Output 4-channel bayer image of shape ``(..., 4, h//2 , w//2)``.

        Has the same dtype and device as input.
    """
    check_rgb(x)
    sensor_alignment = check_sensor_alignment(sensor_alignment)
    h, w = x.shape[-2], x.shape[-1]
    assert h % 2 == w % 2 == 0  # input rgb image must have even-length height and width
    y_list = []
    for r in 0, 1:
        for c in 0, 1:
            color_ind = _color_dict[sensor_alignment[r * 2 + c]]
            y_list.append(x[..., color_ind, r::2, c::2])
    y = torch.stack(y_list, dim=-3)
    return y
