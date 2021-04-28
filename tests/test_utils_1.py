import unittest
from torchimage.utils import NdSpec


class MyTestCase(unittest.TestCase):
    def test_1(self):
        result = NdSpec([[1, 2]], item_shape=[2]).is_item
        expected = False

        self.assertEqual(
            result, expected
        )


if __name__ == '__main__':
    unittest.main()
