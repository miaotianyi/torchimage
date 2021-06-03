import unittest
import torch
import numpy as np
from torchimage.misc import poly1d


class MyTestCase(unittest.TestCase):
    def test_poly1d(self):
        p = np.random.rand(8)
        x = np.random.rand(10, 3, 16, 16)
        expected = np.poly1d(p)(x)
        actual = poly1d(torch.from_numpy(x), p).numpy()
        self.assertLess(np.abs(expected - actual).max(), 1e-12)


if __name__ == '__main__':
    unittest.main()
