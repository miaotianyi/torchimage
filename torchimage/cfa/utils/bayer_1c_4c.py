"""
Conversion between 2 alternative representations of a Bayer array.

| 1-channel representation: ``(..., 1, 2h, 2w)``
| 4-channel representation: ``(..., 4, h, w)`` where the 4 color channels are separated

Notice that the depth dimension (number of color channels) is always on axis ``-3``. Negative indexing is more
compatible with different dataset shapes.

A single bayer array under 1-channel representation is a ``(1, 2h, 2w)`` tessellation like the one below, where
``ABCD`` can be any of the ``RGGB``, ``BGGR``, ``GRBG``, ``GBRG`` sensor alignments:

| ``ABABAB``
| ``CDCDCD``
| ``ABABAB``
| ``CDCDCD``

When converted to 4-channel representation, the same bayer array will be as follows (stack of 4 2D arrays):

| ``AAA BBB CCC DDD``
| ``AAA BBB CCC DDD``

The process is deterministic and fully invertible (doesn't involve any loss of information).

See Also
--------
Demosaic (MatLab): https://www.mathworks.com/help/images/ref/demosaic.html
"""


import torch

from .validation import check_4c_bayer, check_1c_bayer


def bayer_1c_to_4c(x: torch.Tensor):
    """
    Convert 1-channel bayer array to 4-channel bayer array.

    Input shape is automatically validated, so it's not necessary to check again outside of the scope.

    Parameters
    ----------
    x : torch.Tensor
        Input 1-channel Bayer array of shape ``(..., 1, h, w)``

    Returns
    -------
    y : torch.Tensor
        Output 4-channel Bayer array of shape ``(..., 4, h//2, w//2)``

        Has the same dtype and device as the input tensor.
    """
    check_1c_bayer(x)
    channel_list = []  # list of color channels
    for r in 0, 1:
        for c in 0, 1:
            channel_list.append(x[..., r::2, c::2])
    y = torch.cat(channel_list, dim=-3)
    return y


def bayer_4c_to_1c(x: torch.Tensor):
    """
    Convert 4-channel bayer array to 1-channel bayer array.

    Input shape is automatically validated, so it's not necessary to check again outside of the scope.

    Parameters
    ----------
    x : torch.Tensor
        Input 4-channel Bayer array of shape ``(..., 4, h, w)``

    Returns
    -------
    y : torch.Tensor
        Output 1-channel Bayer array of shape ``(..., 1, 2*h, 2*w)``

        Has the same dtype and device as the input tensor.
    """
    check_4c_bayer(x)
    h, w = x.shape[-2], x.shape[-1]
    y = torch.empty(x.shape[:-3] + (1, 2*h, 2*w), dtype=x.dtype, device=x.device)
    channel_index = 0
    for r in 0, 1:
        for c in 0, 1:
            y[..., 0, r::2, c::2] = x[..., channel_index, :, :]
            channel_index += 1
    return y


