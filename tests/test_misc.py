import unittest
import torch
import numpy as np
from itertools import product
from torchimage.misc import poly1d, safe_power


class MyTestCase(unittest.TestCase):
    def test_poly1d(self):
        p = np.random.rand(8)
        x = np.random.rand(10, 3, 16, 16)
        expected = np.poly1d(p)(x)
        actual = poly1d(torch.from_numpy(x), p).numpy()
        self.assertLess(np.abs(expected - actual).max(), 1e-12)

    def test_safe_power_1(self):
        a = torch.tensor(-0.0019923369962678242, requires_grad=True)
        b = 0.1333
        c = safe_power(a, b)
        self.assertTrue(torch.isnan(a ** b))
        self.assertFalse(torch.isnan(c))
        c.backward()
        self.assertFalse(torch.isnan(a.grad))

    def test_safe_power_2(self):
        a = -0.0019923369962678242
        b = torch.tensor(0.1333, requires_grad=True)
        c = safe_power(a, b)
        # self.assertTrue(torch.isnan(a ** b))
        self.assertFalse(torch.isnan(c))
        c.backward()
        self.assertFalse(torch.isnan(b.grad))

    def test_safe_power_3(self):
        a = torch.tensor(-42.389401, requires_grad=True)
        b = -0.1333
        c = safe_power(a, b)
        self.assertFalse(torch.isnan(c))
        c.backward()
        self.assertFalse(torch.isnan(a.grad))

    def test_safe_power_4(self):
        a = torch.tensor([0, 1.5, 0.23], requires_grad=True)
        b = 0.21429
        c = safe_power(a, b).sum()
        c.backward()
        self.assertGreater(a.grad[0], 0)

    def test_safe_power_5(self):
        for b in range(-4, 10, 2):
            a = torch.tensor(-1.2, requires_grad=True)
            c = safe_power(a, b)
            with self.subTest(b=b):
                self.assertGreater(c, 0)
                self.assertEqual(c, a.item() ** b)

    def test_safe_power_6(self):
        def get_func(name):
            if name == "rand":
                return lambda *size: torch.rand(*size)
            elif name == "ones":
                return lambda *size: torch.ones(size)
            elif name == "zeros":
                return lambda *size: torch.zeros(size)

        for f1, f2 in product(["rand", "ones", "zeros"], repeat=2):
            if f1 == "zeros" and f2 == "rand":
                continue
            a, b = get_func(f1)(10, 3, 7), get_func(f2)(7)
            with self.subTest(f1=f1, f2=f2):
                self.assertTrue(torch.all(a ** b - safe_power(a, b) == 0))



if __name__ == '__main__':
    unittest.main()
