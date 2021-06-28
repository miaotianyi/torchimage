import torch
import numpy as np

from .mse import MSE


class PSNR(MSE):
    def __init__(self, max_value=1, eps=0.0, *, reduction="mean"):
        super().__init__(reduction=reduction)
        self.max_value = max_value  # input must be in range [0, max_value]
        # prevent nan gradient in log10
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, reduce_axes=None):
        mse_score = super().forward(y_pred=y_pred, y_true=y_true, reduce_axes=reduce_axes)
        log_mse = torch.log10(mse_score + self.eps if self.eps > 0 else mse_score)
        return 20 * np.log10(self.max_value) - 10 * log_mse

