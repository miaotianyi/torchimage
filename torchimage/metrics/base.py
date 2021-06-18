from abc import abstractmethod

import torch

from ..utils.validation import check_axes


class BaseMetric:
    """
    Base class for all metrics such as PSNR, SSIM, and multi-scale SSIM.

    This base class mostly handles reduction (aggregation) of output
    scores, allowing for different output shapes:
    1. scalar: Useful as a loss in PyTorch neural network training
    2. (n_samples, ): See scores of different data points

    Use keyword axes to denote the "content axes" such as
    depth, height, and width. They will be averaged in the final output.
    If axes is None, all axes will be averaged.

    In SSIM, ``axes`` has special meaning that the averaging kernel
    will be convolved on those axes only (doesn't include batch
    or channel dimensions).


    Compatible with PyTorch, aggregation modes include ``'mean'``, ``'sum'``, and
    ``'none'`` (do not change full output at all).
    """
    @abstractmethod
    def forward_full(self, y_pred: torch.Tensor, y_true: torch.Tensor, axes):
        pass

    def __init__(self, *, reduction="mean"):
        self.reduction = reduction

    def _reduce(self, x: torch.Tensor, axes: tuple):
        if self.reduction == "mean":
            return self._mean(x, axes)
        elif self.reduction == "sum":
            return self._sum(x, axes)
        elif self.reduction == "none":
            return x
        else:
            raise ValueError(f"Unknown reduction mode: {self.reduction}")

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, axes=None):
        assert y_pred.shape == y_true.shape

        axes = check_axes(y_pred, axes)
        # don't need to check axes later

        x = self.forward_full(y_pred=y_pred, y_true=y_true, axes=axes)
        x = self._reduce(x, axes=axes)
        return x

    @staticmethod
    def _mean(x: torch.Tensor, axes):
        return x.mean(dim=axes, keepdim=False)

    @staticmethod
    def _sum(x: torch.Tensor, axes):
        return x.sum(dim=axes, keepdim=False)

