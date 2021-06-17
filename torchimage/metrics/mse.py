import torch

from .base import BaseMetric


class MSE(BaseMetric):
    def forward_full(self, y_pred: torch.Tensor, y_true: torch.Tensor, axes):
        return (y_pred - y_true) ** 2


def mse(image_true, image_test):
    """
    Mean squared error

    Parameters
    ----------
    image_true
    image_test

    Returns
    -------

    """
    # input image must be 3D (c, h, w) or (n, c, h, w)
    # the last 3 dimensions will be reduced
    mse = torch.mean((image_true - image_test) ** 2, dim=[-1, -2, -3])
    return mse
