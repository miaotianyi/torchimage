"""
Customized neural network layer for bayer array convolution
4 distinct filter groups for R, B, GR, and GB
"""
import numpy as np
import torch
from torch import nn

from ..utils.validation import check_sensor_alignment

relative_positions = "R", "B", "GR", "GB"
absolute_positions = "00", "01", "10", "11"


_color_dict = {
    "R": 0,
    "G": 1,
    "B": 2
}


def get_padding_layer(kernel_size, beg_row, beg_col):
    """
    The padding layer accepts a batch of unsqueezed bayer arrays of shape (n_samples, c, height, width)
      and returns the batch where the bayer arrays are padded
      (n_samples, c, height + kernel_size - 2, width + kernel_size - 2)
    The padding layer serves 2 purposes:
    1. Align the center of the convolutional kernel to the target pixels in the 2D bayer array.
      Because the convolutional kernel always starts "sliding" at top left, we can control the offset
      (where the center is/where the kernel begins)
    2. Deal with borders elegantly. The intuition of bilinear interpolation stems from "taking the average of nearby
      pixels" of nearest neighbor algorithms, so when the only available neighbors are on on side, taking just that
      neighbor's value is equivalent to taking the average of two copies.
    """
    offset = kernel_size // 2  # equal to (kernel_size - 1) / 2
    pad_top = offset - beg_row
    pad_bottom = (offset + 1) - (2 - beg_row)
    pad_left = offset - beg_col
    pad_right = (offset + 1) - (2 - beg_col)
    padding_layer = nn.ReflectionPad2d(padding=(pad_left, pad_right, pad_top, pad_bottom))
    return padding_layer


def get_sensor_beg_index(sensor_alignment):
    """
    Input: sensor alignment specification (str) such as "GRBG"
    Output: dict that maps pixel type (i.e. GR for G at R row) to its starting index (i.e. (0, 0)) at mod 2
    """
    ret = {
        "R": divmod(sensor_alignment.index("R"), 2),  # red locations start, at (row index, column index)
        "B": divmod(sensor_alignment.index("B"), 2),  # blue locations start, at (row index, column index)
    }
    ret["GR"] = ret["R"][0], 1 - ret["R"][1]  # green locations in red row
    ret["GB"] = ret["B"][0], 1 - ret["B"][1]  # green locations in blue row
    return ret


class BayerConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias):
        super(BayerConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.bias = bias

        self.conv_dict = nn.ModuleDict()
        self.padding_dict = nn.ModuleDict()

        # initialize padding layers: starting position % 2 tuple -> padding layer
        # padding layers are always bound with its absolute location in an array
        # note that nn.ModuleDict doesn't allow non-string keys
        for pos_abs in absolute_positions:
            self.padding_dict[pos_abs] = get_padding_layer(
                kernel_size=self.kernel_size, beg_row=int(pos_abs[0]), beg_col=int(pos_abs[1])
            )

        # convolution layers are bound with the relative color locations in a bayer array
        # relative positions
        for pos_rel in relative_positions:
            self.conv_dict[pos_rel] = nn.Conv2d(
                in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size,
                stride=2, padding=0, bias=self.bias
            )

    def forward(self, x: torch.Tensor, sensor_alignment: str):
        # x: ..., in_channels, h, w
        # the channel dimension is all that matters
        y = torch.empty(x.shape[:-3] + (self.out_channels,) + x.shape[-2:], dtype=x.dtype, device=x.device)

        sensor_alignment = check_sensor_alignment(sensor_alignment)
        # maps a relative position (GR, R, etc.) to its absolute position tuple (e.g. (0, 1)) under this sensor
        # alignment
        beg_dict = get_sensor_beg_index(sensor_alignment)
        for pos_rel in relative_positions:
            r, c = beg_dict[pos_rel]
            pos_abs = str(r) + str(c)
            conv = self.conv_dict[pos_rel]
            pad = self.padding_dict[pos_abs]
            y[..., r::2, c::2] = conv(pad(x))
        return y


def last_step_demosaic(x: torch.Tensor, y: torch.Tensor, sensor_alignment: str):
    # fill in the original colors
    # x: ..., 1, h, w; the original input bayer
    # y: ..., 2, h, w; the output interpolation from network for the other 2 colors
    # R -> G, B; G -> R, B; B -> R, G
    # We assume that the color channels of y exactly follow such order
    sensor_alignment = check_sensor_alignment(sensor_alignment)
    beg_dict = get_sensor_beg_index(sensor_alignment)
    assert x.shape[:-3] == y.shape[:-3] and x.shape[-2:] == y.shape[-2:]

    ret = torch.empty(x.shape[:-3] + (3,) + x.shape[-2:], dtype=x.dtype, device=x.device)
    for pos_rel in relative_positions:
        color_name = pos_rel[0]
        color_index = _color_dict[color_name]  # GR and GB map to G
        r, c = beg_dict[pos_rel]  # absolute position
        ret[..., [color_index], r::2, c::2] = x[..., :, r::2, c::2]
        color_0_name, color_1_name = "RGB".replace(color_name, "")  # names of other two color channels
        ret[..., [_color_dict[color_0_name]], r::2, c::2] = y[..., [0], r::2, c::2]
        ret[..., [_color_dict[color_1_name]], r::2, c::2] = y[..., [1], r::2, c::2]
    return ret


class PreTrainedBilinearInterpolator(nn.Module):
    def __init__(self, weight_dict: dict):
        super(PreTrainedBilinearInterpolator, self).__init__()
        # this is the backward compatible version of BilinearInterpolator; should have the same effect

        # weight dict: str (e.g. BR, BGR, BGB) -> np array (or nested list as 2d array) of weight
        kernel_size = next(iter(weight_dict.values())).shape[0]
        self.decision_layer = BayerConv2d(in_channels=1, out_channels=2, kernel_size=kernel_size, bias=False)

        for pos_rel in relative_positions:  # GR, GB, B, R
            color_name = pos_rel[0]
            # original weight (nn.Parameter) to be replaced
            old_weight = self.decision_layer.conv_dict[pos_rel].weight
            color_0, color_1 = "RGB".replace(color_name, "")  # names of other two color channels
            numpy_weight = np.stack([
                np.array(weight_dict[color_0 + pos_rel]), np.array(weight_dict[color_1 + pos_rel])
            ])
            new_weight = nn.Parameter(
                torch.from_numpy(numpy_weight).to(dtype=old_weight.dtype, device=old_weight.device).unsqueeze(-3)
            )
            # unsqueeze -3 here: note that the parameter shape is (out channels, in channels, kernel, kernel)
            self.decision_layer.conv_dict[pos_rel].weight = new_weight

    def forward(self, x, sensor_alignment, clamp=1.0):
        # input: ..., 1, h, w; 1-channel bayer tensor; has a trivial channel dimension
        # this is significantly different from the previous input convention (where the trivial dimension is removed)
        # output: ..., 3, h, w; RGB tensor
        y = self.decision_layer(x, sensor_alignment=sensor_alignment)
        return last_step_demosaic(x, y, sensor_alignment).clamp(0.0, clamp)

