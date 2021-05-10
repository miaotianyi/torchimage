import unittest

from itertools import product
import torch
import numpy as np
import pywt
from torchimage.padding.generic_pad import GenericPadNd, _stat_padding_set


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tol = 1e-8
        self.n_trials = 20

    def assertArrayEqual(self, a, b, tol=None, msg=None):
        if tol is None:
            tol = self.tol
        lib = torch if (torch.is_tensor(a) and torch.is_tensor(b)) else np
        self.assertLess(lib.abs(a - b).sum(), tol, msg=f"{a} is not equal to {b}")

    def test_const_padding(self):
        mode = "constant"
        for _ in range(self.n_trials):
            ndim = np.random.randint(1, 6)
            shape = np.random.randint(1, 7, ndim)

            # fine for pad width to exceed original length
            pad_width_scalar = np.random.randint(0, 10)
            pad_width_pair = np.random.randint(0, 10, 2).tolist()
            pad_width_list = np.random.randint(0, 10, (ndim, 2)).tolist()

            constant_values_scalar = np.random.rand()
            constant_values_pair = np.random.rand(2).tolist()
            constant_values_list = np.random.rand(ndim, 2).tolist()

            arr_1 = np.random.rand(*shape)
            arr_2 = torch.from_numpy(arr_1)

            for pw, cv in product([pad_width_scalar, pad_width_pair, pad_width_list],
                                  [constant_values_scalar, constant_values_pair, constant_values_list]):
                with self.subTest(shape=shape, pad_width=pw, constant_values=cv):
                    expected = np.pad(arr_1, pad_width=pw, mode=mode, constant_values=cv)
                    actual = GenericPadNd(pad_width=pw, mode=mode, constant_values=cv)(arr_2).numpy()
                    self.assertArrayEqual(expected, actual)

    def test_stat_padding(self):
        for mode in _stat_padding_set:
            if mode == "median":  # numpy is known to behave differently when it comes to medians
                continue
            for _ in range(self.n_trials):
                ndim = np.random.randint(1, 6)
                shape = np.random.randint(1, 7, ndim)

                # fine for pad width to exceed original length
                pad_width_scalar = np.random.randint(0, 10)
                pad_width_pair = np.random.randint(0, 10, 2).tolist()
                pad_width_list = np.random.randint(0, 10, (ndim, 2)).tolist()

                stat_length_scalar = np.random.randint(1, min(shape) + 1)
                stat_length_pair = np.random.randint(1, min(shape) + 1, 2).tolist()
                stat_length_list = np.random.randint(1, np.tile((shape + 1).reshape(-1, 1), [1, 2])).tolist()

                arr_1 = np.random.rand(*shape)
                arr_2 = torch.from_numpy(arr_1)

                for pw, sl in product([pad_width_scalar, pad_width_pair, pad_width_list],
                                      [stat_length_scalar, stat_length_pair, stat_length_list]):
                    with self.subTest(shape=shape, mode=mode, stat_length=sl):
                        expected = np.pad(arr_1, pad_width=pw, mode=mode, stat_length=sl)
                        actual = GenericPadNd(pad_width=pw, mode=mode, stat_length=sl)(arr_2).numpy()
                        self.assertArrayEqual(expected, actual)

    def test_replicate(self):
        mode_ti = "replicate"
        mode_np = "edge"
        for _ in range(self.n_trials):
            ndim = np.random.randint(1, 6)
            shape = np.random.randint(1, 7, ndim)

            # fine for pad width to exceed original length
            pad_width_scalar = np.random.randint(0, 10)
            pad_width_pair = np.random.randint(0, 10, 2).tolist()
            pad_width_list = np.random.randint(0, 10, (ndim, 2)).tolist()

            arr_1 = np.random.rand(*shape)
            arr_2 = torch.from_numpy(arr_1)

            for pw in [pad_width_scalar, pad_width_pair, pad_width_list]:
                with self.subTest(shape=shape, pad_width=pw):
                    expected = np.pad(arr_1, pad_width=pw, mode=mode_np)
                    actual = GenericPadNd(pad_width=pw, mode=mode_ti)(arr_2).numpy()
                    self.assertArrayEqual(expected, actual)

    def test_smooth_padding(self):
        mode = "smooth"
        for i in range(self.n_trials):
            ndim = np.random.randint(1, 6)
            shape = np.random.randint(2, 7, ndim)

            # fine for pad width to exceed original length
            # we only test compatibility for meaningful values
            # if we disagree with PyWavelet or Numpy, we won't test for equality

            # Important Note: PyWavelets runs into bugs with pad_width=0 at any axis in some cases

            # Note 2: PyWavelets smooth is ill-defined when an axis only has 1 element
            # For instance, pywt.pad([50.0], pad_widths=3, mode="smooth") gives
            # [ 200.  150.  100.   50.    0.  -50. -100.]
            # This is in fact caused by first placing the original array inside a new,
            # larger zero array and then, without checking the boundaries, mistaking
            # the 0 on the right of 50 as if it were a legitimate element. This 0 is used
            # to extrapolate elements on the left, where 100 is used to extrapolate
            # elements on the right

            pad_width_scalar = np.random.randint(1, 10)
            pad_width_pair = np.random.randint(1, 10, 2).tolist()
            pad_width_list = np.random.randint(1, 10, (ndim, 2)).tolist()

            arr_1 = np.random.rand(*shape)
            arr_2 = torch.from_numpy(arr_1)

            for pw in [pad_width_scalar, pad_width_pair, pad_width_list]:
                with self.subTest(shape=shape, pad_width=pw):
                    # print(i)
                    # print("calculating expected")
                    expected = pywt.pad(arr_1, pw, mode=mode)
                    # print("calculating actual")
                    actual = GenericPadNd(pad_width=pw, mode=mode)(arr_2).numpy()
                    # print("Done.")
                    self.assertArrayEqual(expected, actual)

    def test_circular(self):
        for mode_ti, mode_wt in [
            ["circular", "periodic"],
            ["periodize", "periodization"],
            ["symmetric"] * 2,
            ["reflect"] * 2,
            ["antisymmetric"] * 2,
            ["odd_reflect", "antireflect"]
        ]:
            for _ in range(self.n_trials):
                ndim = np.random.randint(1, 6)
                shape = np.random.randint(1, 7, ndim)

                # Note: cannot exceed original length
                # otherwise numpy padding runs into a known bug
                pad_width_scalar = np.random.randint(1, min(shape) + 1)
                pad_width_pair = np.random.randint(1, min(shape) + 1, 2).tolist()
                pad_width_list = np.random.randint(1, np.tile((shape + 1).reshape(-1, 1), [1, 2])).tolist()

                arr_1 = np.random.rand(*shape)
                arr_2 = torch.from_numpy(arr_1)

                for pw in [pad_width_scalar, pad_width_pair, pad_width_list]:
                    with self.subTest(mode=mode_ti, shape=shape, pad_width=pw):
                        print(f"{mode_ti=}, {shape=}, {pw=}")
                        print("calculating expected")
                        expected = pywt.pad(arr_1, pad_widths=pw, mode=mode_wt)
                        print("calculating actual")
                        actual = GenericPadNd(pad_width=pw, mode=mode_ti)(arr_2).numpy()
                        print("Done.")
                        self.assertArrayEqual(expected, actual)

    def test_odd_symmetric(self):
        for _ in range(self.n_trials):
            ndim = np.random.randint(1, 6)
            shape = np.random.randint(1, 7, ndim)

            # Note: cannot exceed original length
            # otherwise numpy padding runs into a known bug
            pad_width_scalar = np.random.randint(1, min(shape) + 1)
            pad_width_pair = np.random.randint(1, min(shape) + 1, 2).tolist()
            pad_width_list = np.random.randint(1, np.tile((shape + 1).reshape(-1, 1), [1, 2])).tolist()

            arr_1 = np.random.rand(*shape)
            arr_2 = torch.from_numpy(arr_1)

            for pw in [pad_width_scalar, pad_width_pair, pad_width_list]:
                with self.subTest(shape=shape, pad_width=pw):
                    print(f"{shape=}, {pw=}")
                    print("calculating expected")
                    expected = np.pad(arr_1, pad_width=pw, mode="symmetric", reflect_type="odd")
                    print("calculating actual")
                    actual = GenericPadNd(pad_width=pw, mode="odd_symmetric")(arr_2).numpy()
                    print("Done.")
                    self.assertArrayEqual(expected, actual)

    def test_linear_ramp(self):
        mode = "linear_ramp"
        for _ in range(self.n_trials):
            ndim = np.random.randint(1, 6)
            shape = np.random.randint(1, 7, ndim)

            # fine for pad width to exceed original length
            pad_width_scalar = np.random.randint(0, 10)
            pad_width_pair = np.random.randint(0, 10, 2).tolist()
            pad_width_list = np.random.randint(0, 10, (ndim, 2)).tolist()

            end_values_scalar = np.random.rand()
            end_values_pair = np.random.rand(2).tolist()
            end_values_list = np.random.rand(ndim, 2).tolist()

            arr_1 = np.random.rand(*shape)
            arr_2 = torch.from_numpy(arr_1)

            for pw, ev in product([pad_width_scalar, pad_width_pair, pad_width_list],
                                  [end_values_scalar, end_values_pair, end_values_list]):
                with self.subTest(shape=shape, pad_width=pw, end_values=ev):
                    expected = np.pad(arr_1, pad_width=pw, mode=mode, end_values=ev)
                    actual = GenericPadNd(pad_width=pw, mode=mode, end_values=ev)(arr_2).numpy()
                    self.assertArrayEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()