import torch

from .base import BaseMetric


class MSE(BaseMetric):
    def forward_full(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        return (y_pred - y_true) ** 2
