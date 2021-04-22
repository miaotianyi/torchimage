"""
Gradient-corrected bilinear interpolation, the algorithm in MATLAB's demosaic function.
Implemented as a collection of (5, 5) filters
"""
import numpy as np

from .bayer_conv_2d import PreTrainedBilinearInterpolator

__all__ = ["gradient_corrected_interpolator"]

# list of optimal wiener filters
wf_list = [None, None, None, None]

# G at R locations and G at B locations
wf_list[0] = np.array([
    [0, 0, -1, 0, 0],
    [0, 0, 2, 0, 0],
    [-1, 2, 4, 2, -1],
    [0, 0, 2, 0, 0],
    [0, 0, -1, 0, 0]
]) / 8

# R at green in R row, B column; B at green in B row, R column
wf_list[1] = np.array([
    [0, 0, 1/2, 0, 0],
    [0, -1, 0, -1, 0],
    [-1, 4, 5, 4, -1],
    [0, -1, 0, -1, 0],
    [0, 0, 1/2, 0, 0]
]) / 8

# R at green in B row, R column; B at green in R row, B column
wf_list[2] = wf_list[1].T

# R at B; B at R
wf_list[3] = np.array([
    [0, 0, -3/2, 0, 0],
    [0, 2, 0, 2, 0],
    [-3/2, 0, 6, 0, -3/2],
    [0, 2, 0, 2, 0],
    [0, 0, -3/2, 0, 0]
]) / 8


# IMPORTANT: wf_dict is required for every instance of generalized bilinear interpolation
# Every item in wf_dict should be a 2D array of kernel weights
# Although the real nn.Parameter needs to be (1, 1, h, w), the unsqueezing operation is conducted later in
# get_conv_layer_from_weights function
wf_dict = {
    "GR": wf_list[0], "GB": wf_list[0],
    "RGR": wf_list[1], "BGB": wf_list[1],
    "RGB": wf_list[2], "BGR": wf_list[2],
    "BR": wf_list[3], "RB": wf_list[3]
}

gradient_corrected_interpolator = PreTrainedBilinearInterpolator(weight_dict=wf_dict)
