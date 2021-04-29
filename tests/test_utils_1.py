import unittest
from torchimage.utils import NdSpec


class MyTestCase(unittest.TestCase):
    def test_1(self):
        obj = NdSpec([[1, 2]], item_shape=[2])
        self.assertFalse(obj.is_item)
        self.assertEqual(obj(-1), [1, 2])

    def test_2(self):
        obj = NdSpec([1, 2], item_shape=[2])
        self.assertTrue(obj.is_item)
        for i in range(1, 10):
            self.assertEqual(obj(-i), [1, 2])

    def test_3(self):
        obj = NdSpec(423, item_shape=[])
        self.assertTrue(obj.is_item)
        for i in range(1, 10):
            self.assertEqual(obj(-i), 423)

    def test_4(self):
        obj = NdSpec(423, item_shape=[2])
        self.assertTrue(obj.is_item)
        for i in range(1, 10):
            self.assertEqual(obj(-i), [423, 423])


if __name__ == '__main__':
    unittest.main()
