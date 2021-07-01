import numpy as np
import torch
from torch import nn

from ..utils.validation import check_axes
from ..pooling import BasePoolNd, AvgPoolNd, GaussianPoolNd


class SSIM(nn.Module):
    def __init__(self, blur: BasePoolNd = "gaussian",
                 padder=None,
                 K1=0.01, K2=0.03,
                 use_sample_covariance=True, crop_border=True):
        """
        Compute the structural similarity index between two images.

        Bigger is better.

        This function is intended to imitate the behavior of its counterpart in scikit-image.

        Parameters
        ----------
        blur : str or BasePoolNd
            A pooling layer that meaningfully takes the mean for each
            sliding window. The user may choose to customize a BasePoolNd
            object.

            If ``"gaussian"``, will use ``GaussianPoolNd(kernel_size=11, sigma=1.5)``,
            which is used in the original paper.

            If ``"mean"``, will use ``AvgPoolNd(kernel_size=11)``

            The pooling layer will be automatically converted to a filtering layer
            with same padding, padding option specified by `padder`, and stride of 1.

        padder : str or Padder
            Padding option.

        K1 : float
            Small constant for algorithm. Default: 0.01

        K2 : float
            Small constant for algorithm. Default: 0.03

        use_sample_covariance : bool
            If True, normalize covariances by N-1 rather than N,
            where N is the number of pixels within the sliding window.

        crop_border : bool
            Whether to ignore the strips of border in the averaging calculation
            across the content axes (height, width, etc.).
            Default: True (as in scikit-image)

            A border pixel's input sliding window contains padded values,
            so their values might suffer from border effects.

        See Also
        --------
        Image Quality Assessment: From Error Visibility to Structural Similarity
        """
        super(SSIM, self).__init__()

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

    def _crop_border(self, full: torch.Tensor, content_axes: tuple):
        if not self.crop_border or self.blur.same_padder is None:
            return full
        idx = [slice(None)] * full.ndim  # new idx slice
        pad_width = self.blur.same_padder.pad_width
        # access last-used padding width
        for i, a in enumerate(content_axes):
            pad_before, pad_after = pad_width[i]
            idx[a] = slice(pad_before, full.shape[a]-pad_after)
        return full[tuple(idx)]

    def _reduce(self, x: torch.Tensor, content_axes: tuple, reduce_axes: tuple):
        return self._crop_border(x, content_axes=content_axes).mean(dim=reduce_axes)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor,
                content_axes=slice(2, None), reduce_axes=slice(1, None), *, full=False):
        """
        Parameters
        ----------
        y_pred : torch.Tensor
            The first input tensor.
            Order doesn't matter because SSIM is symmetric with respect to input images.

        y_true : torch.Tensor
            The second input tensor.

        content_axes : None, int, slice, tuple
            Axes that describe the "content" of an image.
            This includes depth, height, and width but excludes batch or channel dimensions.

        reduce_axes : None, int, slice, tuple
            The final SSIM score will average the full SSIM map across these axes.

            If reduce_axes is None (all axes are reduce axes), the output score
            will be a scalar (useful as a loss function). If reduce_axes doesn't
            include batch axes, then it returns a 1d tensor of SSIM scores for
            every data point.

        full : bool
            Whether to return the full SSIM map as well. Default: False.

        Returns
        -------
        score : torch.Tensor
            The SSIM score tensor where content axes are reduced.

        full_tensor : torch.Tensor
            The full output SSIM with the same shape as input images.
            This argument is only returned when ``full=True``.
        """
        assert y_pred.shape == y_true.shape

        content_axes = check_axes(y_pred, content_axes)
        reduce_axes = check_axes(y_pred, reduce_axes)

        full_tensor = self._ssim_full(y1=y_pred, y2=y_true, axes=content_axes)
        score = self._reduce(full_tensor, content_axes=content_axes, reduce_axes=reduce_axes)
        if full:
            return score, full_tensor
        else:
            return score


class MS_SSIM(SSIM):
    def __init__(self,
                 weights=None, use_prod=True,
                 blur: BasePoolNd = "gaussian",
                 padder=None,
                 K1=0.01, K2=0.03,
                 eps=1e-8,
                 use_sample_covariance=True, crop_border=True
                 ):
        """
        Compute the mean multi-scale structural similarity index between two images.

        The full SSIM matrix is not well-defined in this scenario,
        because there are multiple such matrices with different shapes.
        Therefore, it is not returned and there's no ``full`` parameter.

        Parameters
        ----------
        weights : tuple
            Weight for score maps in each level. Default: ``[0.0448, 0.2856, 0.3001, 0.2363, 0.1333]`` as in original paper.

        use_prod : bool
            If True, uses prod(s^weight for s in ...) instead of sum(s*weight for s in ...),
            where s is the score map. Default: True.

        blur
        padder
        K1
        K2
        eps
        use_sample_covariance
        crop_border

        See Also
        --------
        Multiscale structural similarity for image quality assessment
        https://ece.uwaterloo.ca/~z70wang/publications/msssim.pdf
        """
        
        super().__init__(blur=blur, padder=padder, K1=K1, K2=K2,
                         use_sample_covariance=use_sample_covariance, crop_border=crop_border)

        # multiscale-specific settings
        if weights is None:
            self.weights = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)
        else:
            self.weights = weights

        self.use_prod = use_prod
        self.eps = eps

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

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor,
                content_axes=slice(2, None), reduce_axes=slice(1, None), **kwargs):
        # before score computation, channel axes will be averaged first
        # so the final axes are just the sample dimensions

        # data validation
        assert y_pred.shape == y_true.shape

        # axes
        content_axes = check_axes(y_pred, content_axes)
        reduce_axes = check_axes(y_pred, reduce_axes)

        # content axes are large enough
        self._check_shape_large_enough(y_pred, axes=content_axes)

        final_score = None

        y1, y2 = y_pred, y_true

        for i in range(self.n_levels - 1):  # only last layer uses full l*c*s
            cs_full = self._cs_full(y1=y1, y2=y2, axes=content_axes)  # n, c, d, h, w
            cs_full = self._crop_border(cs_full, content_axes=content_axes)
            cs_score = cs_full.mean(dim=content_axes, keepdim=True)

            if final_score is None:
                if self.use_prod:
                    final_score = cs_score.clamp(self.eps) ** self.weights[i]
                else:
                    final_score = cs_score * self.weights[i]
            else:
                if self.use_prod:
                    final_score *= cs_score.clamp(self.eps) ** self.weights[i]
                else:
                    final_score += cs_score * self.weights[i]

            y1 = self._downsample(y1, axes=content_axes)
            y2 = self._downsample(y2, axes=content_axes)

        ssim_full = self._ssim_full(y1=y1, y2=y2, axes=content_axes)
        ssim_full = self._crop_border(ssim_full, content_axes=content_axes)
        ssim_score = ssim_full.mean(dim=content_axes, keepdim=True)

        if self.use_prod:
            # negative values need to be removed
            final_score *= ssim_score.clamp(self.eps) ** self.weights[-1]
        else:
            final_score += ssim_score * self.weights[-1]

        # the scores are first separated by batches and channels
        # only at the very end are channels aggregated
        final_score = final_score.mean(dim=reduce_axes, keepdim=False)

        return final_score

