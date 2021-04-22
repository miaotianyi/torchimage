import torch
from torch.nn import functional as F

from ..filtering import GenericFilter2d


def _mu_sigma(img1, img2, filter_layer, scaling_factor):
    mu1 = filter_layer(img1)
    mu2 = filter_layer(img2)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = filter_layer(img1 * img1) - mu1_sq
    sigma2_sq = filter_layer(img2 * img2) - mu2_sq
    sigma12 = filter_layer(img1 * img2) - mu1_mu2

    if scaling_factor is not None:
        sigma1_sq *= scaling_factor
        sigma2_sq *= scaling_factor
        sigma12 *= scaling_factor

    return mu1_sq, mu2_sq, mu1_mu2, sigma1_sq, sigma2_sq, sigma12


def _ssim_map(mu1_sq, mu2_sq, mu1_mu2, sigma1_sq, sigma2_sq, sigma12, C1, C2):
    return ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))


def _cs_map(sigma1_sq, sigma2_sq, sigma12, C2):
    # only calculates c * s (out of l, c, s in SSIM)
    return (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)


def _map_mean(score_map, keep_channels, crop_edge, window_size):
    # given a score map of shape [n, c, ...]
    # returns an average score of shape [n, c] (keep channels) or [n]
    dims = list(range(2 if keep_channels else 1, score_map.ndim))
    if crop_edge:
        pad = window_size // 2
        idx = tuple([...] + [slice(pad, -pad) for _ in range(score_map.ndim - 2)])
        score_map = score_map[idx]
    score = torch.mean(score_map, dim=dims)
    return score


def ssim(img1, img2, window_size=11,
         gaussian_weights=True, sigma=1.5,
         K1=0.01, K2=0.03, use_sample_covariance=True,
         padding_mode="replicate", padding_value=0.0,
         full=False, keep_channels=False, crop_edge=True):
    """
    Compute the mean structural similarity index between two images

    This function is intended to imitate the behavior of its counterpart in scikit-image.

    Parameters
    ----------
    img1, img2 : torch.Tensor
        Images of shape (n, c, h, w). Values are in range [0, 1]

    window_size : int
        The side-length of the sliding window used in comparison (kernel size) Default: 11 (as in original paper)

        Sequence of ints will be supported later.

    gaussian_weights : bool
        If True, each patch has its mean and variance spatially weighted by a normalized Gaussian kernel. Otherwise
        average pooling is used.

        Default: ``True`` (as in the original paper)

    sigma : float
        Standard deviation for Gaussian kernel. Default: ``1.5`` (as in original paper).

    K1, K2 : float
        Algorithm parameter, small constant.

    use_sample_covariance : bool
        If True, normalize covariances by N-1 rather than, N where N is the number of pixels within the sliding window.

    padding_mode : str
        Padding mode (using PyTorch's naming convention). Default: ``replicate`` (as in scikit-image).

    padding_value : float
        Padding value, only used when ``padding_mode`` is ``constant``.

    full : bool
        If True, also return the full structural similarity image. (Default: False)

    keep_channels : bool
        If True, return SSIM scores separately for each color channel. Otherwise for each image,
        SSIM scores will be averaged across the color channels.

    crop_edge : bool
        If True, will ignore filter radius strip around edges to avoid edge effects. Default: True (as in scikit-image)

    Returns
    -------
    mssim : torch.Tensor
        Mean ssim of shape (n,) where n is the number of samples.

        If keep_channels is True, mssim will be of shape (n, c) where c is the number of channels.

    S : torch.Tensor
        Full SSIM image of shape (n, c, h, w)

    See Also
    --------
    Image Quality Assessment: From Error Visibility to Structural Similarity
    """
    # images must be in nchw format
    if gaussian_weights:  # use gaussian weighted averaging
        filter_layer = GenericFilter2d("gaussian", kernel_size=window_size, padding_mode=padding_mode, padding_value=padding_value, sigma=sigma)
    else:  # use simple averaging
        filter_layer = GenericFilter2d("mean", kernel_size=window_size, padding_mode=padding_mode, padding_value=padding_value)

    scaling_factor = (n_pixels := window_size ** 2) / (n_pixels - 1) if use_sample_covariance else None

    mu1_sq, mu2_sq, mu1_mu2, sigma1_sq, sigma2_sq, sigma12 = _mu_sigma(img1, img2, filter_layer, scaling_factor)

    C1 = K1 ** 2
    C2 = K2 ** 2

    ssim_map = _ssim_map(mu1_sq, mu2_sq, mu1_mu2, sigma1_sq, sigma2_sq, sigma12, C1, C2)

    ssim_score = _map_mean(ssim_map, keep_channels, crop_edge, window_size)

    if full:
        return ssim_score, ssim_map
    else:
        return ssim_score


def multi_ssim(img1, img2, window_size=11,
               weights=None, use_prod=True,
               gaussian_weights=True, sigma=1.5,
               K1=0.01, K2=0.03, use_sample_covariance=True,
               padding_mode="replicate", padding_value=0.0,
               keep_channels=False, crop_edge=True):
    """
    Compute the mean multi-scale structural similarity index between two images.

    The full SSIM matrix is not well-defined in this scenario as there are multiple such matrices, most of which
    being actually cs without the l component. Therefore, it is not returned and there's no ``full`` parameter.

    Parameters
    ----------
    img1, img2 : torch.Tensor
        Images of shape (n, c, h, w). Values are in range [0, 1]

    window_size : int
        The side-length of the sliding window used in comparison (kernel size) Default: 11 (as in original paper)

        Sequence of ints will be supported later.

    weights : list
        Weight for score maps in each level. Default: ``[0.0448, 0.2856, 0.3001, 0.2363, 0.1333]`` as in original paper.

    use_prod : bool
        If True, uses prod(s^weight for s in ...) instead of sum(s*weight for s in ...), where s is the score map.
        Default: True.

    gaussian_weights, sigma, K1, K2, use_sample_covariance, padding_mode, padding_value, keep_channels, crop_edge
        For other parameters, refer to the documentation for ssim.

    Returns
    -------
    overall_mssim : torch.Tensor
        Mean multi-scale ssim of shape (n,) where n is the number of samples.

        If keep_channels is True, mssim will be of shape (n, c) where c is the number of channels.

    See Also
    --------
    Multiscale structural similarity for image quality assessment
    https://ece.uwaterloo.ca/~z70wang/publications/msssim.pdf
    """
    # multiscale-specific settings
    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    n_levels = len(weights)

    # data validation
    assert img1.shape == img2.shape
    assert min(img1.shape[2:]) / (2 ** (n_levels - 1)) > window_size

    # downsampling method
    def downsample(x):
        return F.avg_pool2d(x, kernel_size=2)

    if gaussian_weights:  # use gaussian weighted averaging
        filter_layer = GenericFilter2d("gaussian", kernel_size=window_size, padding_mode=padding_mode, padding_value=padding_value, sigma=sigma)
    else:  # use simple averaging
        filter_layer = GenericFilter2d("mean", kernel_size=window_size, padding_mode=padding_mode, padding_value=padding_value)

    C1 = K1 ** 2
    C2 = K2 ** 2

    scaling_factor = (n_pixels := window_size ** 2) / (n_pixels - 1) if use_sample_covariance else None

    overall_mssim = 1 if use_prod else 0  # product vs sum

    for i in range(n_levels - 1):  # only last layer uses full l*c*s
        mu1_sq, mu2_sq, mu1_mu2, sigma1_sq, sigma2_sq, sigma12 = _mu_sigma(img1, img2, filter_layer, scaling_factor)
        cs_map = _cs_map(sigma1_sq, sigma2_sq, sigma12, C2)
        mcs = _map_mean(cs_map, keep_channels, crop_edge, window_size)  # mean cs of shape [n,] or [n, c]

        if use_prod:
            overall_mssim *= mcs ** weights[i]
        else:
            overall_mssim += mcs * weights[i]
        img1, img2 = downsample(img1), downsample(img2)

    mu1_sq, mu2_sq, mu1_mu2, sigma1_sq, sigma2_sq, sigma12 = _mu_sigma(img1, img2, filter_layer, scaling_factor)
    ssim_map_last = _ssim_map(mu1_sq, mu2_sq, mu1_mu2, sigma1_sq, sigma2_sq, sigma12, C1, C2)
    mssim_last = _map_mean(ssim_map_last, keep_channels, crop_edge, window_size)

    if use_prod:
        overall_mssim *= mssim_last ** weights[-1]
    else:
        overall_mssim += mssim_last * weights[-1]

    return overall_mssim
