import unittest

import numpy as np
import torch
from torch.nn import MSELoss

from torchimage.metrics.mse import MSE
from torchimage.metrics.psnr import PSNR
from torchimage.metrics.ssim_new import SSIM, MS_SSIM

from torchimage.padding import Padder

# from torchimage.metrics import ssim as my_ssim, multiscale_ssim as my_multiscale_ssim, psnr as my_psnr
# from examples.metrics import ssim, msssim, psnr

from skimage.metrics import structural_similarity, peak_signal_noise_ratio


class MyTestCase(unittest.TestCase):
    # def test_ssim(self):
    #     n = 1000
    #     y1 = torch.rand(1, 1, 128, 128)
    #     y2 = torch.rand(1, 1, 128, 128)
    #     # expected = structural_similarity(y1.squeeze().numpy(), y2.squeeze().numpy(), data_range=1, gaussian_weights=True, use_sample_covariance=False, K1=0.01, K2=0.03)
    #     my_score, my_result = my_ssim(y1, y2, window_size=11, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, padding_mode="constant", K1=0.01, K2=0.03, full=True, crop_edge=False)
    #     my_result = my_result.numpy()
    #
    #     your_result = ssim(y1, y2, window_size=11, size_average=False).numpy()
    #     your_score = ssim(y1, y2, window_size=11, size_average=True)
    #     print(np.abs(my_result - your_result).max())
    #     print(my_score, your_score)

    def test_mse(self):
        for reduction in ['mean', 'sum', 'none']:
            m1 = MSE(reduction=reduction)
            m2 = MSELoss(reduction=reduction)
            shape = np.random.randint(1, 10, size=np.random.randint(1, 8))
            y1, y2 = torch.rand(*shape), torch.rand(*shape)
            r1 = m1.forward(y1, y2)
            r2 = m2.forward(y1, y2)
            with self.subTest(reduction=reduction):
                self.assertTrue(torch.all(r1 - r2 == 0))

    def test_psnr(self):
        shape = np.random.randint(1, 10, size=np.random.randint(1, 8))
        y1, y2 = torch.rand(*shape, dtype=torch.float64), torch.rand(*shape, dtype=torch.float64)

        actual = PSNR(max_value=1, reduction='mean').forward(y1, y2, reduce_axes=None).item()
        expected = peak_signal_noise_ratio(y1.numpy(), y2.numpy(), data_range=1)
        self.assertLess(np.abs(actual - expected), 1e-10)

    def test_ssim_1(self):
        # TODO: padding is too slow in high dimensions (e.g. 5)
        shape = np.random.randint(15, 20, size=np.random.randint(1, 6))

        y1, y2 = torch.rand(*shape, dtype=torch.float64), torch.rand(*shape, dtype=torch.float64)

        win_size = 11

        for gaussian_weights, blur in [(True, "gaussian"), (False, "mean")]:
            for multichannel, reduce_axes, content_axes in [(False, None, None), (True, None, slice(0, -1))]:
                expected_score, expected_full = structural_similarity(
                    y1.numpy(), y2.numpy(), win_size=win_size, gradient=False, data_range=1,
                    multichannel=multichannel, gaussian_weights=gaussian_weights, full=True,
                    use_sample_covariance=False,
                )
                actual_score, actual_full = SSIM(
                    blur=blur, padder="symmetric", K1=0.01, K2=0.03,
                    use_sample_covariance=False, crop_border=True).forward(
                    y1, y2, content_axes=content_axes, reduce_axes=reduce_axes, full=True
                )
                actual_score = actual_score.item()
                actual_full = actual_full.numpy()
                with self.subTest(gaussian_weights=gaussian_weights, multichannel=multichannel):
                    self.assertLess(np.abs(actual_full - expected_full).max(), 1e-13)
                    self.assertLess(abs(expected_score - actual_score), 1e-14)

    # @unittest.skip
    # def test_multi_ssim(self):
    #     # from IQA_pytorch import MS_SSIM as their_ms_ssim, SSIM as their_ssim
    #     from examples.multi_ssim import MS_SSIM as their_ms_ssim
    #     from torchimage.metrics.ssim import multiscale_ssim as old_ssim
    #     from torchimage.random import add_gauss_noise
    #     y1 = torch.rand(1, 3, 256, 256, dtype=torch.float64)
    #     # y2 = add_gauss_noise(y1, sigma=0.08).clamp(0, 1)
    #     y2 = torch.rand(1, 3, 256, 256, dtype=torch.float64)
    #
    #     expected = their_ms_ssim(data_range=1).forward(y1, y2)
    #     actual = MS_SSIM(use_prod=True, padder=None, use_sample_covariance=True, crop_border=True).forward(
    #         y1, y2, content_axes=(2, 3), reduce_axes=(1, 2, 3))[0]
    #     print(expected.item())
    #     print(actual.item())


if __name__ == '__main__':
    unittest.main()
