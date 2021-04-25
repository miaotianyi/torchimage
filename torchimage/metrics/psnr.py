import torch
from .mse import mse


def psnr(image_true, image_test, data_range=1):
    """
    Peak signal-to-noise ratio

    Parameters
    ----------
    image_true
    image_test
    data_range

    Returns
    -------

    """
    # input image must be 3D (c, h, w) or (n, c, h, w)
    # the last 3 dimensions will be reduced
    score = 20 * torch.log10(data_range / torch.sqrt(
        mse(image_true, image_test)
    ))
    return score
