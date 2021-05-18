import unittest
import torch
from torch import nn
from torch.nn import functional as F
from torchimage.cfa import BayerConv2d


class MyTestCase(unittest.TestCase):
    def test_unshuffle(self):
        in_channels = 7
        sensor_alignment = "RGGB"
        layer1 = BayerConv2d(in_channels=in_channels, out_channels=10, kernel_size=5, bias=True)
        for x in layer1.parameters():
            nn.init.normal_(x)
        a = torch.rand(10, in_channels, 80, 90)
        expected = layer1(a, sensor_alignment=sensor_alignment)
        result = layer1.forward_unshuffle(a, sensor_alignment=sensor_alignment)
        actual = F.pixel_shuffle(result, upscale_factor=2)
        self.assertLess(torch.abs(expected-actual).sum(), 1e-5)


if __name__ == '__main__':
    unittest.main()
