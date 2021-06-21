import numpy as np
import torch

from .base import BaseMetric
from ..utils.validation import check_axes
from ..padding import Padder
from ..pooling import BasePoolNd, AvgPoolNd, GaussianPoolNd


class SSIM:
    def __init__(self, blur: BasePoolNd = "gaussian",
                 padder: Padder = Padder(mode="replicate"),
                 K1=0.01, K2=0.03,
                 use_sample_covariance=True, crop_border=True):
        if blur == "gaussian":
            self.blur = GaussianPoolNd(kernel_size=11, sigma=1.5).to_filter(padder)
        elif blur == "mean":
            self.blur = AvgPoolNd(kernel_size=11).to_filter(padder)
        else:
            self.blur = blur.to_filter(padder)

        self.K1 = K1
        self.K2 = K2
        self.use_sample_covariance = use_sample_covariance
        self.crop_border = crop_border

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

        sigma1_sq = self.blur.forward(y1 * y1, axes=axes) - mu1_sq
        sigma2_sq = self.blur.forward(y2 * y2, axes=axes) - mu2_sq
        sigma12 = self.blur.forward(y1 * y2, axes=axes) - mu1_mu2

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

    def _crop_border(self, full: torch.Tensor, axes: tuple):
        if not self.crop_border:
            return full
        idx = [slice(None)] * full.ndim  # new idx slice
        pad_width = self.blur.same_padder.pad_width
        # access last-used padding width
        for i, a in enumerate(axes):
            pad_before, pad_after = pad_width[i]
            idx[i] = slice(pad_before, full.shape[a]-pad_after)
        return full[tuple(idx)]

    def forward_score(self, x: torch.Tensor, content_axes: tuple, avg_axes: tuple):
        return self._crop_border(x, axes=content_axes).mean(dim=avg_axes)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, axes=slice(2, None), channel_axes=(1,)):
        assert y_pred.shape == y_true.shape

        axes = check_axes(y_pred, axes)
        channel_axes = check_axes(y_pred, channel_axes)
        avg_axes = channel_axes + axes

        full = self._ssim_full(y1=y_pred, y2=y_true, axes=axes)
        score = self.forward_score(full, content_axes=axes, avg_axes=avg_axes)
        return score, full


class MultiSSIM(SSIM):
    def __init__(self,
                 weights=None, use_prod=True,
                 blur: BasePoolNd = "gaussian",
                 padder: Padder = Padder(mode="replicate"),
                 K1=0.01, K2=0.03,
                 use_sample_covariance=True, crop_border=True
                 ):
        super().__init__(blur=blur, padder=padder, K1=K1, K2=K2,
                         use_sample_covariance=use_sample_covariance, crop_border=crop_border)

        # multiscale-specific settings
        if weights is None:
            self.weights = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)
        else:
            self.weights = weights

        self.use_prod = use_prod

    @staticmethod
    def _downsample(x: torch.Tensor, axes: tuple):
        downsample_layer = AvgPoolNd(kernel_size=2, stride=2)
        return downsample_layer.forward(x, axes=axes)

    @property
    def n_levels(self):
        return len(self.weights)

    def _cs_full(self, y1, y2, axes):
        mu1_sq, mu2_sq, mu1_mu2, sigma1_sq, sigma2_sq, sigma12 = self._mu_sigma(y1=y1, y2=y2, axes=axes)

        C2 = self.K2 ** 2
        # only calculates c * s (out of l, c, s in SSIM)
        return (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)

    def _check_shape_large_enough(self, x: torch.Tensor, axes: tuple):
        factor = (2 ** (self.n_levels - 1))

        for i, a in enumerate(axes):
            assert x.shape[a] / factor > self.blur.kernel_size[i]

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, axes=slice(2, None), channel_axes=(1, )):
        # before score computation, channel axes will be averaged first
        # so the final axes are just the sample dimensions

        # data validation
        assert y_pred.shape == y_true.shape

        # axes
        axes = check_axes(y_pred, axes)
        channel_axes = check_axes(y_pred, channel_axes)
        avg_axes = channel_axes + axes

        # content axes are large enough
        self._check_shape_large_enough(y_pred, axes=axes)

        final_score = None

        y1, y2 = y_pred, y_true

        for i in range(self.n_levels - 1):  # only last layer uses full l*c*s
            cs_full = self._cs_full(y1=y1, y2=y2, axes=axes)  # n, c, d, h, w
            cs_full = self._crop_border(cs_full, axes=axes)
            cs_score = cs_full.mean(dim=avg_axes, keepdim=False)

            if final_score is None:
                if self.use_prod:
                    final_score = cs_score ** self.weights[i]
                else:
                    final_score = cs_score * self.weights[i]
            else:
                if self.use_prod:
                    final_score *= cs_score ** self.weights[i]
                else:
                    final_score += cs_score * self.weights[i]

            y1 = self._downsample(y1, axes=axes)
            y2 = self._downsample(y2, axes=axes)

        ssim_full = self._ssim_full(y1=y1, y2=y2, axes=axes)
        ssim_full = self._crop_border(ssim_full, axes=axes)
        ssim_score = ssim_full.mean(dim=avg_axes, keepdim=False)

        if self.use_prod:
            final_score *= ssim_score ** self.weights[-1]
        else:
            final_score += ssim_score * self.weights[-1]

        return final_score







