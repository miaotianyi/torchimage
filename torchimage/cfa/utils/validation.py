"""
This module contains various validation tools for bayer and rgb images.

The validation tools deal with generalized nchw format: `(number of samples, color channels, height, width)`. The number
of samples can span multiple dimensions, such as `(number of batches, number of samples in each batch, ...)`.
"""

import torch


def check_sensor_alignment(sensor_alignment):
    """
    Check if the input is a valid bayer sensor alignment pattern.

    There are only 4 possible bayer sensor alignment patterns: `GBRG`, `GRBG`, `BGGR`, `RGGB`.

    Parameters
    ----------
    sensor_alignment : str
        Should be one of  `"GBRG"`, `"GRBG"`, `"BGGR"`, and `"RGGB"`.
        Lower case letters are allowed but discouraged.

    Returns
    -------
    sensor_alignment: str
        The sensor alignment itself if the validation is successful. Lower case letters will be automatically
        converted to upper case.

    Raises
    ------
    AssertionError
        Whenever the input fails a validation criterion. The assertion line itself should be helpful enough.
    """
    assert isinstance(sensor_alignment, str), f"sensor_alignment should be string. " \
                                              f"Got {type(sensor_alignment)} instance {sensor_alignment} instead."
    assert len(sensor_alignment) == 4, f"sensor_alignment should have length 4. " \
                                       f"Got {sensor_alignment} of length {len(sensor_alignment)} instead."
    sensor_alignment = sensor_alignment.upper()
    assert sensor_alignment in (all_sa := ("GBRG", "GRBG", "BGGR", "RGGB")),\
        f"sensor_alignment should be one of {all_sa}. Got {sensor_alignment} instead."
    return sensor_alignment


def check_1c_bayer(x: torch.Tensor):
    """
    Check if the input consists of 1-channel bayer arrays

    Parameters
    ----------
    x : torch.Tensor
        Should be 1-channel Bayer array of shape ``(..., 1, h, w)``

    Raises
    ------
    AssertionError
        Whenever the input fails a validation criterion. The assertion line itself should be helpful enough.
    """
    assert isinstance(x, torch.Tensor), f"Input {x=} should be torch.Tensor. Got {type(x)} instance {x} instead."
    assert x.ndim >= 3, f"Shape must at least contain (1, height, width) as the last 3 dimensions." \
                        f"Got {x.ndim}-dimensional tensor of shape {x.shape} instead."
    assert x.shape[-3] == 1, f"A 1-channel Bayer array must have exactly 1 channel." \
                             f"Got {x.shape[-3]} channel(s) in shape {x.shape} instead."
    assert x.shape[-2] % 2 == x.shape[-1] % 2 == 0, "A 1-channel Bayer array's height and width " \
                                                    f"must be divisible by 2. Got {x.shape=} instead."


def check_4c_bayer(x: torch.Tensor):
    """
    Check if the input consists of 4-channel bayer arrays

    Parameters
    ----------
    x : torch.Tensor
        Should be 4-channel Bayer array of shape ``(..., 4, h, w)``

    Raises
    ------
    AssertionError
        Whenever the input fails a validation criterion. The assertion line itself should be helpful enough.
    """
    assert isinstance(x, torch.Tensor), f"Input {x=} should be torch.Tensor. Got {type(x)} instance {x} instead."
    assert x.ndim >= 3, f"Shape must at least contain (4, height, width) as the last 3 dimensions." \
                        f"Got {x.ndim}-dimensional tensor of shape {x.shape} instead."
    assert x.shape[-3] == 4, f"A 4-channel Bayer array must have exactly 4 channels." \
                             f"Got {x.shape[-3]} channel(s) in shape {x.shape} instead."


def check_rgb(x: torch.Tensor):
    """
    Check if the input is a valid RGB image (or image batch) in generalized nchw format.

    Parameters
    ----------
    x : torch.Tensor
        Should be a 3-channel array of shape ``(..., 3, h, w)``.

        The color channels should follow the exact order of Red, Green, and Blue,
        but this order can't and won't be verified.

    Raises
    ------
    AssertionError
        Whenever the input fails a validation criterion. The assertion line itself should be helpful enough.
    """
    assert isinstance(x, torch.Tensor), f"Input {x=} should be torch.Tensor. Got {type(x)} instance {x} instead."
    assert x.ndim >= 3, f"Shape must at least contain (3, height, width) as the last 3 dimensions." \
                        f"Got {x.ndim}-dimensional tensor of shape {x.shape} instead."
    assert x.shape[-3] == 3, f"An RGB image must have exactly 3 channels." \
                             f"Got {x.shape[-3]} channel(s) in shape {x.shape} instead."
