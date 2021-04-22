from .psnr import psnr
from .mse import mse
from .ssim import ssim, multi_ssim

peak_signal_to_noise = psnr
mean_squared_error = mse
structural_similarity = ssim
multiscale_structural_similarity = multi_ssim

__all__ = [
    "psnr", "peak_signal_to_noise",
    "mse", "mean_squared_error",
    "ssim", "structural_similarity",
    "multi_ssim", "multiscale_structural_similarity"
]
