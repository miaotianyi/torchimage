import unittest
from torchimage.pooling.base import SeparablePoolNd
from torchimage.filtering.base import SeparableFilterNd
from scipy import ndimage
from skimage import filters


class MyTestCase(unittest.TestCase):
    def test_average(self):
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
