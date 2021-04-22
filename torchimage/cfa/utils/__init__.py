"""
This submodule contains tools for converting between 1-channel and 4-channel bayer images,
converting RGB images to bayer images, and validating input tensors or keywords for such functionalities.
"""

from .rgb_to_bayer import rgb_to_1c_bayer, rgb_to_4c_bayer
from .bayer_1c_4c import bayer_1c_to_4c, bayer_4c_to_1c
from .validation import check_sensor_alignment, check_1c_bayer, check_4c_bayer, check_rgb
from .bayer_to_rgb import downsample_1c_bayer_to_rgb, downsample_4c_bayer_to_rgb

__all__ = [
    "rgb_to_1c_bayer", "rgb_to_4c_bayer",
    "bayer_1c_to_4c", "bayer_4c_to_1c",
    "check_sensor_alignment", "check_1c_bayer", "check_4c_bayer", "check_rgb",
    "downsample_1c_bayer_to_rgb", "downsample_4c_bayer_to_rgb"
]
