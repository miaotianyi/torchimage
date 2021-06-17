import unittest

import numpy as np
import torch
from torch.nn import MSELoss

from torchimage.metrics.mse import MSE
from torchimage.metrics.psnr import PSNR

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

        actual = PSNR(max_value=1, reduction='mean').forward(y1, y2, axes=None).item()
        expected = peak_signal_noise_ratio(y1.numpy(), y2.numpy(), data_range=1)
        self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()