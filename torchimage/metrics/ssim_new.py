import numpy as np
import torch

from .base import BaseMetric
from ..padding import Padder
from ..pooling import BasePoolNd, AvgPoolNd, GaussianPoolNd


class SSIM(BaseMetric):
    def __init__(self, blur: BasePoolNd = "gaussian",
                 padder: Padder = Padder(mode="replicate"),
                 K1=0.01, K2=0.03,
                 use_sample_covariance=True, crop_edge=True,
                 reduction="mean"):
        super().__init__(reduction=reduction)

        if blur == "gaussian":
            self.blur = GaussianPoolNd(kernel_size=11, sigma=1.5).to_filter(padder)
        elif blur == "mean":
            self.blur = AvgPoolNd(kernel_size=11).to_filter(padder)
        else:
            self.blur = blur.to_filter(padder)

        self.K1 = K1
        self.K2 = K2
        self.use_sample_covariance = use_sample_covariance
        self.crop_edge = crop_edge

    def _scale_factor(self, axes: tuple):
        """
        Compute a scaling factor to multiply the output by,
        if we use sample covariance (``n_pixels - 1`` as divisor) instead
        of population covariance (``n_pixels`` as divisor)
        in the computation of any variance/covariance,
        such as sigma1, sigma2, sigma12.

        Parameters
        ----------
        axes : tuple of int
            Content axes to use in the input image

        Returns
        -------
        scaling_factor : float
            Scaling factor for any sigma
        """
        if self.use_sample_covariance:
            n_pixels = np.prod([self.blur.kernel_size[i] for i in range(len(axes))])
            return n_pixels / (n_pixels - 1)
        else:
            return 1

    def _mu_sigma(self, y1, y2, axes):
        mu1 = self.blur.forward(y1, axes=axes)
        mu2 = self.blur.forward(y2, axes=axes)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = self.blur.forward(y1 * y1) - mu1_sq
        sigma2_sq = self.blur.forward(y2 * y2) - mu2_sq
        sigma12 = self.blur.forward(y1 * y2) - mu1_mu2

        scaling_factor = self._scale_factor(axes=axes)
        if scaling_factor != 1:
            sigma1_sq = sigma1_sq * scaling_factor
            sigma2_sq = sigma2_sq * scaling_factor
            sigma12 = sigma12 * scaling_factor

        return mu1_sq, mu2_sq, mu1_mu2, sigma1_sq, sigma2_sq, sigma12

    def _ssim_full(self, y1, y2, axes):
        mu1_sq, mu2_sq, mu1_mu2, sigma1_sq, sigma2_sq, sigma12 = self._mu_sigma(y1=y1, y2=y2, axes=axes)

        C1 = self.K1 ** 2
        C2 = self.K2 ** 2
        return ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    def forward_full(self, y_pred: torch.Tensor, y_true: torch.Tensor, axes):
        ssim_full = self._ssim_full(y1=y_pred, y2=y_true, axes=axes)

        # ssim_score = _map_mean(ssim_map, keep_channels, crop_edge, window_size)

        return ssim_full

class MultiSSIM(BaseMetric):
    pass
